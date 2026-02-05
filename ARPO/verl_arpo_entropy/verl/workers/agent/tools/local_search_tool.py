"""
Local Search Tool for ARPO - 使用 Search-R1 本地检索服务器

这个工具类是一个适配器，将 Search-R1 的检索服务集成到 ARPO 的 workers 工具系统中。
它使用 verl.tools.utils.search_r1_like_utils 中的函数来调用本地检索服务器。
"""

import json
import logging
from typing import Optional

from verl.workers.agent.tools.base_tool import BaseTool
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

logger = logging.getLogger(__name__)


class LocalSearchTool(BaseTool):
    """
    本地检索工具，使用 Search-R1 检索服务器。
    
    这个工具通过 HTTP API 调用本地检索服务器（如 Search-R1 的 retrieval_server.py），
    支持 BM25 和 Dense Retrieval（FAISS）。
    
    配置示例:
        retrieval_service_url: "http://localhost:8000/retrieve"
        topk: 10
        timeout: 30
    """
    
    @property
    def name(self) -> str:
        """工具名称"""
        return "local_search"
    
    @property
    def trigger_tag(self) -> str:
        """触发该工具的标签"""
        return "search"
    
    def __init__(
        self,
        retrieval_service_url: str,
        topk: int = 10,
        timeout: int = 30
    ):
        """
        初始化本地检索工具。
        
        Args:
            retrieval_service_url: 检索服务器 URL，例如 "http://localhost:8000/retrieve"
            topk: 返回 top-k 个检索结果（默认: 10）
            timeout: 请求超时时间（秒，默认: 30）
        """
        if not retrieval_service_url:
            raise ValueError("retrieval_service_url must be provided")
        
        self.retrieval_service_url = retrieval_service_url
        self.topk = topk
        self.timeout = timeout
        
        logger.info(
            f"Initialized LocalSearchTool with "
            f"retrieval_service_url={retrieval_service_url}, "
            f"topk={topk}, timeout={timeout}"
        )
    
    def execute(self, content: str, **kwargs) -> str:
        """
        执行搜索操作。
        
        Args:
            content: 搜索查询字符串，或查询列表
            **kwargs: 其他参数（暂未使用）
        
        Returns:
            格式化后的检索结果字符串
        """
        # 处理输入：支持字符串或列表
        if isinstance(content, str):
            query_list = [content]
        elif isinstance(content, list):
            query_list = content
        else:
            query_list = [str(content)]
        
        try:
            # 调用检索服务
            result_text, metadata = perform_single_search_batch(
                retrieval_service_url=self.retrieval_service_url,
                query_list=query_list,
                topk=self.topk,
                timeout=self.timeout
            )
            
            # 解析 JSON 结果
            try:
                result_dict = json.loads(result_text)
                search_result = result_dict.get("result", "No search results found.")
                
                # 如果是列表（多个查询），合并结果
                if isinstance(search_result, list):
                    search_result = "\n---\n".join(search_result)
                
                logger.debug(f"Search successful: {len(query_list)} queries, status={metadata.get('status')}")
                return search_result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse search result JSON: {e}, raw result: {result_text[:200]}")
                return f"Search error: Failed to parse result. Raw: {result_text[:200]}"
                
        except Exception as e:
            logger.error(f"Search execution failed: {e}", exc_info=True)
            return f"Search error: {str(e)}"


# 为了兼容性，也可以创建一个别名
SearchTool = LocalSearchTool


def main():
    """简单的测试函数，测试本地搜索工具"""
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试配置 - 根据文档使用 http://localhost:8000/retrieve
    retrieval_service_url = "http://localhost:8000/retrieve"
    topk = 10
    timeout = 30
    
    print("=" * 60)
    print("测试 LocalSearchTool")
    print("=" * 60)
    print(f"检索服务器 URL: {retrieval_service_url}")
    print(f"TopK: {topk}")
    print(f"超时时间: {timeout} 秒")
    print()
    
    try:
        # 初始化工具
        print("1. 初始化 LocalSearchTool...")
        tool = LocalSearchTool(
            retrieval_service_url=retrieval_service_url,
            topk=topk,
            timeout=timeout
        )
        print(f"   ✓ 工具初始化成功")
        print(f"   - 工具名称: {tool.name}")
        print(f"   - 触发标签: {tool.trigger_tag}")
        print()
        
        # 测试 1: 单个查询（参考文档格式）
        print("2. 测试单个查询...")
        print(f"   查询: 'Python'")
        result1 = tool.execute("Python")
        print(f"   ✓ 搜索完成")
        print(f"   结果长度: {len(result1)} 字符")
        print(f"   结果预览: {result1[:200]}...")
        print()
        
        # 测试 2: 批量查询（参考文档格式）
        print("3. 测试批量查询...")
        queries = ["Python", "Machine Learning", "Neural Networks"]
        print(f"   查询列表: {queries}")
        result2 = tool.execute(queries)
        print(f"   ✓ 批量搜索完成")
        print(f"   结果长度: {len(result2)} 字符")
        print(f"   结果预览: {result2[:300]}...")
        print()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except ValueError as e:
        print(f"❌ 配置错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
