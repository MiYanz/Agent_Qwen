# DuckDuckGo 服务的薄启动入口：在代理环境中补充 DNS 失败回退。
# 只替换上游地址校验所用的 socket 引用；公网校验仍由上游服务执行，正文也仍由它获取。
"""在原 DuckDuckGo MCP 入口增加代理环境的 DNS 失败回退，不替换其 URL 防护。"""

import ipaddress
import os
import socket
import sys
import threading
import time

import httpx


# 模拟 socket 模块供上游地址校验使用；只重写 getaddrinfo，其余属性转给标准库。
# 它不会替换 HTTP 客户端实际使用的全局 socket，也不会跳过上游公网地址判定。
class DNSFallbackSocket:
    """仅替换上游校验模块的 socket 引用，不修改进程全局 socket。"""

    # proxy 是已有代理地址；缓存只存域名解析结果，锁保护不同解析线程对缓存的访问。
    def __init__(self, proxy):
        self.proxy = proxy
        self._cache = {}
        self._lock = threading.Lock()

    # 未重写的 socket 属性继续使用标准库，包括 gaierror 异常类型，保持上游捕获逻辑兼容。
    def __getattr__(self, name):
        return getattr(socket, name)

    # 保持标准库解析接口的参数和返回结构；优先缓存或系统 DNS，系统失败后才尝试代理 DoH。
    # 返回全部解析地址供上游检查，不能只保留公网结果而掩盖同时存在的私网记录。
    def getaddrinfo(self, host, port, family=0, type=0, proto=0, flags=0):
        # 上游通过 asyncio.to_thread 调用此同步接口，网络等待不会阻塞 MCP 事件循环。
        # 同一域名的 IPv4 与 IPv6 查询分开缓存；端口不影响 DNS 地址，但稍后会用于包装 socket 元组。
        key = (host, family)
        with self._lock:
            cached = self._cache.get(key)
            addresses = cached[1] if cached and cached[0] > time.monotonic() else None
        if addresses is None:
            try:
                return socket.getaddrinfo(host, port, family, type, proto, flags)
            # 只对系统解析失败启用备用路径；普通成功结果不会额外传给第三方解析器。
            except socket.gaierror as original:
                if not self.proxy or not isinstance(host, str) or flags & socket.AI_NUMERICHOST:
                    raise
                try:
                    addresses, ttl = self._resolve(host, family)
                except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
                    raise socket.gaierror(socket.EAI_NONAME, "system DNS and proxy DNS-over-HTTPS failed") from exc
                if not addresses:
                    raise original
                with self._lock:
                    # 有界、短期缓存；不保留 URL、正文或 Prompt。
                    if len(self._cache) >= 64:
                        self._cache.clear()
                    self._cache[key] = (time.monotonic() + ttl, addresses)
                print("[DNS fallback] public resolver used; original SSRF guard remains enabled", file=sys.stderr, flush=True)

        # 把备用 IP 用 AI_NUMERICHOST 转为标准 socket 返回结构，不再为这些数字地址发起域名查询。
        infos = []
        for address in addresses:
            infos.extend(socket.getaddrinfo(address, port, family, type, proto, flags | socket.AI_NUMERICHOST))
        # 不在此过滤私网地址：全部交还上游原始校验，任何非公网结果都应使请求被拒绝。
        return infos

    # 通过固定 Cloudflare HTTPS 入口查询 A/AAAA；family 决定地址族。
    # 返回地址列表及不超过 60 秒的 TTL；响应失败、地址不合法或地址族不符均不当作成功。
    def _resolve(self, host, family):
        # DNS 记录类型 1 是 A，28 是 AAAA；未指定地址族时两类都查询，不能忽略潜在非公网记录。
        query_types = (1,) if family == socket.AF_INET else (28,) if family == socket.AF_INET6 else (1, 28)
        addresses = []
        ttls = [60]
        # 固定公网 IP 入口不依赖已失效的系统 DNS；请求仍经现有代理且验证 TLS。
        with httpx.Client(proxy=self.proxy, trust_env=False, timeout=6, follow_redirects=False) as client:
            for query_type in query_types:
                response = client.get("https://1.1.1.1/dns-query",
                                      params={"name": host, "type": query_type},
                                      headers={"Accept": "application/dns-json"})
                response.raise_for_status()
                payload = response.json()
                if payload.get("Status") != 0:
                    raise ValueError("DNS resolver returned a failure")
                for answer in payload.get("Answer", []):
                    if answer.get("type") != query_type:
                        continue
                    # 无效记录不得传给底层解析器；CNAME 由公共解析器递归解析。
                    address = ipaddress.ip_address(answer["data"])
                    if address.version != (4 if query_type == 1 else 6):
                        raise ValueError("DNS record address family mismatch")
                    addresses.append(str(address))
                    ttls.append(max(0, int(answer.get("TTL", 0))))
        # TTL 取所有记录与 60 秒上限的最小值；TTL 为 0 时缓存立即过期，不人为延长记录有效期。
        return addresses, min(ttls)


# 在 uv 已准备好的工具环境中导入原服务；有代理时替换其校验模块引用，再调用原 main。
# 导入本包装器本身不会启动服务；只有脚本入口才运行这里。
def main():
    from duckduckgo_mcp_server import server

    proxy = (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
             or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy"))
    if proxy:
        # 仅替换地址校验所用的解析器。原 URL/端口/协议/私网及逐跳重定向检查全部保留。
        server.socket = DNSFallbackSocket(proxy)
    server.main()


if __name__ == "__main__":
    main()
