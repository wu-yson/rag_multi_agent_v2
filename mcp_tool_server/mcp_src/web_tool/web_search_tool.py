"""轻量级网络搜索工具：直接请求 Bing 搜索页并解析结果（国内网络可直连），无需 API Key

Bing 对无 cookie 的脚本请求偶尔会返回"品牌泛页"降级结果（如官网关于页/商城分类/云服务），
本模块会对结果做降级识别，并自动换入口/清洗搜索词重试，尽量返回与搜索词相关的真实结果。
"""

import re
import time

import requests
from lxml import etree
from lxml import html as lh

from mcp_src.utils.logger import log

# 默认返回结果条数，子 agent 调用时可通过参数覆盖
DEFAULT_MAX_RESULTS = 5

# Bing 抓取请求头，模拟浏览器，降低被拦截概率
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
}
_TIMEOUT = 15

# 重试前等待秒数，降低触发 Bing 软限流的概率
_DEGRADED_SLEEP = 1.5

# 命中结果里不一定出现的"意图/类别泛词"：降级重试时会从搜索词里去掉，提高命中率
_WEAK_TERMS = (
    "参数", "价格", "报价", "售价", "多少钱", "发布", "上市", "最新", "新款",
    "手机", "折叠屏", "折叠", "机型", "系列", "配置", "规格", "图片", "评测",
    "介绍", "怎么样", "是什么", "有哪些", "哪款", "详情", "推荐", "对比",
    "2026", "2025", "2024",
)

# 型号/数字/字母等"强特征词"的正则（如 18、fold），用于降级识别
_STRONG_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# 查询词分隔符：空格与常见中英文标点（用于把中文查询切成核心词）
_TERM_SEPARATOR_RE = re.compile(r"[\s,，、;；:：/\\|+&()（）\[\]【】<>《》“”‘’]+")


def _build_bing_params(query: str, region: str, max_results: int) -> dict:
    """把 region（如 cn-zh / us-en）转成 Bing 的地区/语言参数。"""
    country, lang = region.lower().split("-", 1)
    mkt = f"{lang}-{country.upper()}"
    setlang = "zh-hans" if lang == "zh" else lang
    return {
        "q": query,
        "mkt": mkt,
        "setlang": setlang,
        "count": str(max_results),
    }


def _build_rss_params(query: str, region: str) -> dict:
    """Bing RSS 接口参数（与 HTML 接口共用 mkt 地区）。"""
    country, lang = region.lower().split("-", 1)
    mkt = f"{lang}-{country.upper()}"
    return {"q": query, "format": "rss", "mkt": mkt}


def _extract_results(html_text: str, max_results: int) -> list:
    """从 Bing 搜索结果页解析 b_algo 列表，提取标题/链接/摘要。"""
    doc = lh.fromstring(html_text)
    items = doc.xpath("//li[contains(@class, 'b_algo')]")
    results = []
    for li in items:
        anchors = li.xpath(".//h2/a")
        if not anchors:
            continue
        anchor = anchors[0]
        href = anchor.get("href", "").strip()
        title = " ".join("".join(anchor.itertext()).split())
        snippet = " ".join("".join(li.xpath(".//p//text()")).split())
        if not href:
            continue
        results.append({"title": title or "无标题", "href": href, "body": snippet})
        if len(results) >= max_results:
            break
    return results


def _extract_rss_results(xml_text: str, max_results: int) -> list:
    """从 Bing RSS 接口解析 item 列表，提取标题/链接/摘要。"""
    doc = etree.fromstring(xml_text.encode("utf-8"))
    results = []
    for item in doc.xpath("//item"):
        title = " ".join("".join(item.xpath(".//title//text()")).split())
        href = "".join(item.xpath(".//link//text()")).strip()
        snippet = " ".join("".join(item.xpath(".//description//text()")).split())
        if not href:
            continue
        results.append({"title": title or "无标题", "href": href, "body": snippet})
        if len(results) >= max_results:
            break
    return results


def _request_bing(
    query: str, region: str, max_results: int,
    host: str = "www.bing.com", rss: bool = False,
) -> list:
    """向指定 Bing 入口发一次请求并解析为统一的结果列表。"""
    url = f"https://{host}/search"
    params = _build_rss_params(query, region) if rss else _build_bing_params(query, region, max_results)
    resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()
    if rss:
        return _extract_rss_results(resp.text, max_results)
    return _extract_results(resp.text, max_results)


def _query_strong_tokens(query: str) -> list:
    """提取搜索词里的强特征词（型号/数字/字母，如 18、fold），用于降级识别。"""
    return [token.lower() for token in _STRONG_TOKEN_RE.findall(query) if len(token) >= 2]


def _strip_weak_terms(text: str) -> str:
    """去掉参数/价格/最新/推荐等泛词，只保留核心词。"""
    cleaned = text
    for word in sorted(_WEAK_TERMS, key=len, reverse=True):
        cleaned = cleaned.replace(word, " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _query_core_terms(query: str) -> list:
    """提取查询核心词（中英文通用，用于泛页识别）。

    先去掉泛词，再按空格/标点切分成片段，保留长度 >= 2 的词。
    例："工程资料管理 最新规范 要求 方法" -> ["工程资料管理", "规范", "要求", "方法"]
    """
    cleaned = _strip_weak_terms(query) or query
    terms = []
    for segment in _TERM_SEPARATOR_RE.split(cleaned):
        segment = segment.strip().lower()
        if len(segment) >= 2:
            terms.append(segment)
    return terms


def _term_hit(term: str, blob: str) -> bool:
    """判断核心词是否在结果里出现过。

    短词要求整词命中；长词（长词组/长句，>= 8 字）改用二元组覆盖率判断，
    避免结果只覆盖词的一部分就被当成"命中"。
    """
    if term in blob:
        return True
    if len(term) >= 8:
        grams = [term[i:i + 2] for i in range(len(term) - 1)]
        if grams:
            hits = sum(1 for gram in grams if gram in blob)
            return hits * 2 >= len(grams)
    return False


def _is_degraded(query: str, results: list) -> bool:
    """判断结果是否为搜索引擎的"泛页"降级结果（中英文查询都适用）。

    规则（满足任一条即判为降级）：
    1) 查询里的型号/数字/字母等强特征词，一个都没在结果里出现；
    2) 查询里的长核心词（>= 4 字，如"工程资料管理"）一个都没在结果里出现。

    Bing 有时会把查询退化成第一个词（如"工程资料管理"->"工程"），返回一批
    与查询无关的通用页面，这种情况需要换入口或换检索词重试。
    """
    strong_tokens = _query_strong_tokens(query)
    terms = _query_core_terms(query)
    if not strong_tokens and not terms:
        return False
    blob = " ".join(
        f"{item.get('title', '')} {item.get('href', '')} {item.get('body', '')}".lower()
        for item in results
    )
    if strong_tokens and not any(token in blob for token in strong_tokens):
        return True
    key_terms = [term for term in terms if len(term) >= 4] or terms
    return not any(_term_hit(term, blob) for term in key_terms)


def _clean_query(query: str) -> str:
    """去掉参数/价格/发布/最新等泛词，保留核心词（用于降级重试）。"""
    return _strip_weak_terms(query) or query

def _format_results(raw_results: list) -> str:
    """把结果列表拼接成返回文本（格式保持不变）。"""
    parts = []
    for i, item in enumerate(raw_results, start=1):
        title = item.get("title", "无标题")
        url = item.get("href", "")
        snippet = item.get("body", "")
        parts.append(f"[{i}] {title}\n    链接: {url}\n    摘要: {snippet}")
    return "\n\n".join(parts)


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    region: str = "cn-zh",
) -> str:
    """在互联网上搜索并返回结构化的搜索结果。

    直接请求 Bing 搜索页并用 lxml 解析（国内网络可直连），无需 API Key，
    适合子 agent 快速获取网页信息。

    说明：Bing 对无 cookie 的脚本请求偶尔会返回品牌泛页降级结果，本函数会
    自动换入口（cn.bing.com / RSS）并清洗搜索词重试，尽量返回真实相关结果。

    Args:
        query: 搜索关键词或问题。
        max_results: 最多返回的结果条数，默认 5 条。
        region: 搜索区域，默认 cn-zh（中文区域），可改为 us-en、wt-wt（全球）等。

    Returns:
        拼接后的搜索结果文本，每条包含标题、链接和摘要。
    """
    if not query or not query.strip():
        return "[ERROR] 搜索关键词不能为空"
    if max_results <= 0:
        return "[ERROR] max_results 必须大于 0"

    query = query.strip()
    log.info(f" [WebSearch] 开始搜索: query={query!r}, max_results={max_results}, region={region}")

    cleaned_query = _clean_query(query)
    attempts = [
        {"host": "www.bing.com", "rss": False, "q": query},
        {"host": "cn.bing.com", "rss": False, "q": cleaned_query},
        {"host": "www.bing.com", "rss": True, "q": query},
    ]

    last_error = None
    degraded_results = None
    completed = 0
    empty_count = 0

    for index, attempt in enumerate(attempts):
        if index > 0:
            time.sleep(_DEGRADED_SLEEP)
        try:
            raw_results = _request_bing(
                attempt["q"], region, max_results,
                host=attempt["host"], rss=attempt["rss"],
            )
            last_error = None
        except Exception as e:
            last_error = e
            log.error(f" [WebSearch] 第 {index + 1} 次请求失败: {e}")
            break

        completed += 1
        if not raw_results:
            empty_count += 1
            log.info(f" [WebSearch] 第 {index + 1} 次未搜索到结果: query={attempt['q']!r}")
            continue

        if _is_degraded(query, raw_results):
            if degraded_results is None:
                degraded_results = raw_results
            log.warning(
                f" [WebSearch] 第 {index + 1} 次疑似降级泛页，准备换入口重试: "
                f"query={attempt['q']!r}, 命中条数={len(raw_results)}"
            )
            continue

        log.info(
            f" [WebSearch] 搜索完成: query={attempt['q']!r}, "
            f"入口={attempt['host']}/{'rss' if attempt['rss'] else 'html'}, "
            f"返回 {len(raw_results)} 条结果"
        )
        return _format_results(raw_results)

    # 所有尝试都没有拿到"干净"结果，按优先级兜底返回
    if degraded_results is not None:
        titles = " / ".join(item.get("title", "无标题") for item in degraded_results[:3])
        log.warning(
            f" [WebSearch] 多次重试后仍判定为泛页，已丢弃 {len(degraded_results)} 条结果: {titles}"
        )
        return (
            f"[INFO] 未搜索到与「{query}」相关的可靠结果\n"
            f"[提示] 搜索入口返回的是与该查询无关的通用页面（搜索引擎降级结果或被识别为机器人），"
            f"已丢弃，请勿据此编造内容；可换用更具体的检索词重试，或基于其他来源回答。"
        )

    if completed > 0 and empty_count == completed:
        return f"[INFO] 未搜索到与「{query}」相关的结果"
    if last_error is not None:
        return f"[ERROR] 网络搜索异常: {last_error}"
    return f"[INFO] 未搜索到与「{query}」相关的结果"
