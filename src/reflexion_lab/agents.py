from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import time
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .ollama_runtime import OllamaRuntimeConfig, OllamaLLM, OllamaActor, OllamaEvaluator, OllamaReflector
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        for attempt_id in range(1, self.max_attempts + 1):
            answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge = evaluator(example, answer)
            # Estimate token count based on attempt number and agent type
            token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
            # Estimate latency in milliseconds based on attempt number and agent type
            latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            if judge.score == 1:
                traces.append(trace)
                break
            
            # TODO: Học viên triển khai logic Reflexion tại đây
            # 1. Kiểm tra nếu agent_type là 'reflexion' và chưa hết số lần attempt
            # 2. Gọi hàm reflector để lấy nội dung reflection
            # 3. Cập nhật reflection_memory để Actor dùng cho lần sau
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection = reflector(example, attempt_id, judge)
                reflections.append(reflection)
                reflection_memory.append(f"Attempt {attempt_id}: {reflection.failure_reason} Lesson: {reflection.lesson} Strategy: {reflection.next_strategy}")
            traces.append(trace)
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)


# Ollama-based agents using local LLM
@dataclass
class OllamaAgent:
    """Base class for Ollama-based agents with local LLM"""
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:latest"
    temperature: float = 0.7
    
    def __post_init__(self):
        """Initialize Ollama components"""
        self.config = OllamaRuntimeConfig(
            base_url=self.ollama_base_url,
            model=self.ollama_model,
            temperature=self.temperature,
        )
        self.llm = OllamaLLM(self.config)
        self.actor = OllamaActor(self.llm)
        self.evaluator = OllamaEvaluator(self.llm)
        self.reflector = OllamaReflector(self.llm)
    
    def run(self, example: QAExample) -> RunRecord:
        """Run example using Ollama LLM components with orchestration"""
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        
        for attempt_id in range(1, self.max_attempts + 1):
            try:
                # Actor generates answer using ReAct reasoning
                answer, act_latency = self.actor.act(example, reflection_memory)
                
                # Evaluator scores the answer
                judge, eval_latency = self.evaluator.evaluate(example, answer)
                
                # Accumulate latencies
                total_latency = act_latency + eval_latency
                
                # Estimate token count based on attempt number and agent type
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                
                # Create trace for this attempt
                trace = AttemptTrace(
                    attempt_id=attempt_id,
                    answer=answer,
                    score=judge.score,
                    reason=judge.reason,
                    token_estimate=token_estimate,
                    latency_ms=total_latency,
                )
                
                final_answer = answer
                final_score = judge.score
                traces.append(trace)
                
                # If correct, stop attempting
                if judge.score == 1:
                    break
                
                # Reflector generates reflection for incorrect answers (only in reflexion mode)
                if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                    reflection, reflect_latency = self.reflector.reflect(
                        example, attempt_id, answer, judge
                    )
                    reflections.append(reflection)
                    reflection_memory.append(
                        f"Attempt {attempt_id}: {reflection.failure_reason} "
                        f"Lesson: {reflection.lesson} Strategy: {reflection.next_strategy}"
                    )
                    
            except Exception as e:
                # Handle errors gracefully
                error_msg = f"Error generating answer: {str(e)}"
                final_answer = error_msg
                final_score = 0
                
                # Estimate token count even for errors
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                
                trace = AttemptTrace(
                    attempt_id=attempt_id,
                    answer=error_msg,
                    score=0,
                    reason=str(e),
                    token_estimate=token_estimate,
                    latency_ms=0,
                )
                traces.append(trace)
                break
        
        # Calculate totals
        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none" if final_score == 1 else "wrong_final_answer"
        
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class OllamaReActAgent(OllamaAgent):
    """ReAct agent using Ollama local LLM"""
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "phi3:latest",
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            agent_type="react",
            max_attempts=1,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            temperature=temperature,
        )


class OllamaReflexionAgent(OllamaAgent):
    """Reflexion agent using Ollama local LLM with reflection mechanism"""
    def __init__(
        self,
        max_attempts: int = 3,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "phi3:latest",
        temperature: float = 0.1,
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            temperature=temperature,
        )
