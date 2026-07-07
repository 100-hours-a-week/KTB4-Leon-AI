from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import Generator, create_generator
from .retriever import KnowledgeIndex, Source


Route = Literal["search_knowledge", "list_capabilities", "list_documents"]


class AgentState(TypedDict, total=False):
    question: str
    top_k: int
    route: Route
    route_reason: str
    tool_context: str
    sources: list[Source]
    tools_used: list[str]
    answer: str
    provider: str
    model: str
    rewrite_count: int
    needs_rewrite: bool
    steps: list[str]


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    answer: str
    route: str
    tools_used: list[str]
    sources: list[Source]
    provider: str
    model: str
    steps: list[str]


class LangGraphRAGAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        generator: Generator | None = None,
        index: KnowledgeIndex | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        if not 0 <= self.settings.chunk_overlap < self.settings.chunk_size:
            raise ValueError("chunk_overlap은 0 이상 chunk_size 미만이어야 합니다.")
        self.generator = generator or create_generator(self.settings)
        self.index = index or KnowledgeIndex(
            self.settings.knowledge_dir,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.graph = self._build_graph()

    @staticmethod
    def _heuristic_route(question: str) -> Route:
        lowered = question.lower()
        if any(term in lowered for term in ("문서 수", "몇 개", "인덱스 상태")):
            return "list_documents"
        if any(term in lowered for term in ("기능", "할 수", "엔드포인트")):
            return "list_capabilities"
        return "search_knowledge"

    def _plan(self, state: AgentState) -> AgentState:
        prompt = f"""질문을 처리할 도구 하나를 고르세요.

- search_knowledge: 챗봇, RAG, 모델, 학습 데이터, API 사용법에 관한 문서 검색
- list_capabilities: Agent가 제공하는 기능과 REST API 안내
- list_documents: 현재 인덱싱된 문서 수와 파일 목록

질문: {state['question']}

JSON만 출력: {{"route":"도구명","reason":"선택 이유"}}"""
        generation = self.generator.generate(
            prompt,
            system="ROUTE_DECISION: 사용자의 질문에 맞는 도구를 선택하는 계획자입니다.",
            max_tokens=120,
            temperature=0.0,
        )
        route = self._heuristic_route(state["question"])
        reason = "규칙 기반 대체 경로"
        try:
            match = re.search(r"\{.*\}", generation.text, re.DOTALL)
            payload = json.loads(match.group(0) if match else generation.text)
            candidate = str(payload.get("route", ""))
            if candidate in {
                "search_knowledge",
                "list_capabilities",
                "list_documents",
            }:
                route = candidate  # type: ignore[assignment]
                reason = str(payload.get("reason", "LLM 도구 선택"))
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        return {
            "route": route,
            "route_reason": reason,
            "provider": generation.provider,
            "model": generation.model,
            "steps": [*state.get("steps", []), f"plan:{route}"],
        }

    @staticmethod
    def _route(state: AgentState) -> Route:
        return state["route"]

    def _search_knowledge(self, state: AgentState) -> AgentState:
        sources = self.index.search(state["question"], state["top_k"])
        if sources:
            context = "\n\n".join(
                f"[출처 {index}: {source.source}]\n{source.text}"
                for index, source in enumerate(sources, start=1)
            )
        else:
            context = "검색된 문서가 없습니다."
        return {
            "tool_context": context,
            "sources": sources,
            "tools_used": ["search_knowledge"],
            "steps": [*state.get("steps", []), "tool:search_knowledge"],
        }

    def _list_capabilities(self, state: AgentState) -> AgentState:
        text = (
            "이 Agent는 지식 문서 검색, 현재 인덱스 상태 확인, 기능 및 API 안내를 "
            "제공한다. 주요 REST API는 POST /api/agent/query, "
            "POST /api/documents/reindex, GET /health, GET /api/graph이다."
        )
        source = Source("agent_capabilities", "agent_capabilities#0", 1.0, text)
        return {
            "tool_context": f"[출처 1: agent_capabilities]\n{text}",
            "sources": [source],
            "tools_used": ["list_capabilities"],
            "steps": [*state.get("steps", []), "tool:list_capabilities"],
        }

    def _list_documents(self, state: AgentState) -> AgentState:
        stats = self.index.stats()
        text = (
            f"현재 {stats['documents']}개 문서와 {stats['chunks']}개 청크가 "
            f"인덱싱되어 있다. 파일 목록: {', '.join(stats['sources'])}"
        )
        source = Source("agent_index", "agent_index#0", 1.0, text)
        return {
            "tool_context": f"[출처 1: agent_index]\n{text}",
            "sources": [source],
            "tools_used": ["list_documents"],
            "steps": [*state.get("steps", []), "tool:list_documents"],
        }

    def _generate_answer(self, state: AgentState) -> AgentState:
        prompt = f"""사용자 질문: {state['question']}

도구 실행 결과:
{state['tool_context']}

도구 결과만 근거로 한국어로 직접 답하세요."""
        generation = self.generator.generate(
            prompt,
            system=(
                "당신은 LangGraph Agent의 답변 노드입니다. 도구 결과 밖의 사실을 "
                "추측하지 말고 근거 문장 끝에 [출처 N]을 표시하세요."
            ),
            max_tokens=700,
            temperature=0.1,
        )
        return {
            "answer": generation.text.strip(),
            "provider": generation.provider,
            "model": generation.model,
            "steps": [*state.get("steps", []), "generate"],
        }

    @staticmethod
    def _verify(state: AgentState) -> AgentState:
        has_sources = bool(state.get("sources"))
        has_citation = "[출처" in state.get("answer", "")
        needs_rewrite = (
            has_sources
            and not has_citation
            and state.get("rewrite_count", 0) < 1
        )
        return {
            "needs_rewrite": needs_rewrite,
            "steps": [
                *state.get("steps", []),
                "verify:rewrite" if needs_rewrite else "verify:passed",
            ],
        }

    @staticmethod
    def _after_verify(state: AgentState) -> Literal["rewrite", "end"]:
        return "rewrite" if state.get("needs_rewrite") else "end"

    def _rewrite(self, state: AgentState) -> AgentState:
        prompt = f"""기존 답변: {state['answer']}

도구 실행 결과:
{state['tool_context']}

내용을 바꾸지 말고 올바른 [출처 N] 표기를 추가하세요."""
        generation = self.generator.generate(
            prompt,
            system="CITATION_REWRITE: 근거 출처가 누락된 답변을 한 번 수정합니다.",
            max_tokens=700,
            temperature=0.0,
        )
        return {
            "answer": generation.text.strip(),
            "provider": generation.provider,
            "model": generation.model,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
            "steps": [*state.get("steps", []), "rewrite"],
        }

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("plan", self._plan)
        builder.add_node("search_knowledge", self._search_knowledge)
        builder.add_node("list_capabilities", self._list_capabilities)
        builder.add_node("list_documents", self._list_documents)
        builder.add_node("generate", self._generate_answer)
        builder.add_node("verify", self._verify)
        builder.add_node("rewrite", self._rewrite)

        builder.add_edge(START, "plan")
        builder.add_conditional_edges(
            "plan",
            self._route,
            {
                "search_knowledge": "search_knowledge",
                "list_capabilities": "list_capabilities",
                "list_documents": "list_documents",
            },
        )
        for tool_node in (
            "search_knowledge",
            "list_capabilities",
            "list_documents",
        ):
            builder.add_edge(tool_node, "generate")
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges(
            "verify",
            self._after_verify,
            {"rewrite": "rewrite", "end": END},
        )
        builder.add_edge("rewrite", "verify")
        return builder.compile(name="week8_langgraph_rag_agent")

    def query(self, question: str, top_k: int | None = None) -> AgentAnswer:
        question = question.strip()
        if not question:
            raise ValueError("질문이 비어 있습니다.")
        state = self.graph.invoke(
            {
                "question": question,
                "top_k": top_k or self.settings.default_top_k,
                "rewrite_count": 0,
                "steps": [],
            }
        )
        return AgentAnswer(
            question=question,
            answer=str(state["answer"]),
            route=str(state["route"]),
            tools_used=list(state.get("tools_used", [])),
            sources=list(state.get("sources", [])),
            provider=str(state.get("provider", "unknown")),
            model=str(state.get("model", "unknown")),
            steps=list(state.get("steps", [])),
        )

    def reindex(self) -> dict[str, int | list[str]]:
        return self.index.reindex()

    def stats(self) -> dict:
        provider = getattr(self.generator, "active_provider", None)
        if provider is None:
            provider = getattr(self.generator, "provider", "configured")
        return {
            **self.index.stats(),
            "framework": "LangGraph StateGraph",
            "active_provider": provider,
            "graph_nodes": [
                "plan",
                "search_knowledge",
                "list_capabilities",
                "list_documents",
                "generate",
                "verify",
                "rewrite",
            ],
        }

    @staticmethod
    def graph_definition() -> dict[str, list]:
        return {
            "nodes": [
                "plan",
                "search_knowledge",
                "list_capabilities",
                "list_documents",
                "generate",
                "verify",
                "rewrite",
            ],
            "edges": [
                ["START", "plan"],
                ["plan", "selected_tool"],
                ["selected_tool", "generate"],
                ["generate", "verify"],
                ["verify", "END|rewrite"],
                ["rewrite", "verify"],
            ],
        }
