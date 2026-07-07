import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx

from langgraph_agent.app import app
from langgraph_agent.config import Settings
from langgraph_agent.graph import LangGraphRAGAgent
from langgraph_agent.llm import Generation, OfflineGenerator


def build_agent(directory: Path) -> LangGraphRAGAgent:
    return LangGraphRAGAgent(
        Settings(
            knowledge_dir=directory,
            primary_provider="offline",
            chunk_size=300,
            chunk_overlap=40,
        ),
        generator=OfflineGenerator(),
    )


def test_state_graph_searches_knowledge(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text(
        "문서는 knowledge 폴더에 추가하고 재인덱싱 API를 호출한다.",
        encoding="utf-8",
    )
    agent = build_agent(tmp_path)

    result = agent.query("문서를 어디에 추가하나요?", top_k=2)

    assert result.route == "search_knowledge"
    assert result.tools_used == ["search_knowledge"]
    assert result.sources[0].source == "guide.md"
    assert "[출처 1]" in result.answer
    assert result.steps[-1] == "verify:passed"


def test_agent_routes_to_index_tool(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("RAG 테스트 문서", encoding="utf-8")
    agent = build_agent(tmp_path)

    result = agent.query("현재 문서는 몇 개야?")

    assert result.route == "list_documents"
    assert result.tools_used == ["list_documents"]
    assert "1개 문서" in result.answer


def test_agent_routes_to_capabilities_tool(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("RAG 테스트 문서", encoding="utf-8")
    agent = build_agent(tmp_path)

    result = agent.query("이 Agent는 어떤 기능을 할 수 있어?")

    assert result.route == "list_capabilities"
    assert result.tools_used == ["list_capabilities"]
    assert "/api/agent/query" in result.answer


def test_agent_api(tmp_path: Path) -> None:
    (tmp_path / "guide.txt").write_text(
        "RAG는 문서를 검색해 답변한다.", encoding="utf-8"
    )
    agent = build_agent(tmp_path)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/api/agent/query",
                json={"question": "RAG는 무엇인가요?", "top_k": 1},
            )

    with patch("langgraph_agent.app.get_agent", return_value=agent):
        response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json()["route"] == "search_knowledge"
    assert response.json()["steps"][0].startswith("plan:")


def test_graph_definition_contains_rewrite_loop(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("RAG 테스트 문서", encoding="utf-8")
    agent = build_agent(tmp_path)

    definition = agent.graph_definition()

    assert "verify" in definition["nodes"]
    assert ["rewrite", "verify"] in definition["edges"]


def test_missing_citation_runs_rewrite_loop(tmp_path: Path) -> None:
    class MissingCitationGenerator:
        def generate(
            self,
            prompt: str,
            *,
            system: str,
            max_tokens: int = 600,
            temperature: float = 0.1,
        ) -> Generation:
            if "ROUTE_DECISION" in system:
                return Generation(
                    '{"route":"search_knowledge","reason":"문서 검색"}',
                    "fake",
                    "fake-model",
                )
            if "CITATION_REWRITE" in system:
                return Generation("근거 답변 [출처 1]", "fake", "fake-model")
            return Generation("근거 답변", "fake", "fake-model")

    (tmp_path / "guide.md").write_text("RAG 근거 문서", encoding="utf-8")
    agent = LangGraphRAGAgent(
        Settings(knowledge_dir=tmp_path, primary_provider="offline"),
        generator=MissingCitationGenerator(),
    )

    result = agent.query("RAG 근거는 무엇인가요?")

    assert result.answer == "근거 답변 [출처 1]"
    assert "rewrite" in result.steps
    assert result.steps[-1] == "verify:passed"
