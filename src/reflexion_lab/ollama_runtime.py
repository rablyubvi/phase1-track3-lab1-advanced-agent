"""
Ollama Runtime for ReAct and Reflexion Agents
Implements local LLM integration using Ollama with scoring and reflection logic.
"""

from __future__ import annotations
import json
import requests
import time
from typing import Optional
from .schemas import QAExample, JudgeResult, ReflectionEntry, AttemptTrace, RunRecord
from .utils import normalize_answer
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM


class OllamaRuntimeConfig:
    """Configuration for Ollama runtime"""
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model="phi3:latest",
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int = 300,
        max_retries: int = 2,
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.max_retries = max_retries


class OllamaLLM:
    """Wrapper for Ollama API calls"""
    
    def __init__(self, config: OllamaRuntimeConfig):
        self.config = config
        self.api_url = f"{config.base_url}/api/generate"
        
    def call(
        self,
        prompt: str,
        system: str = "",
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Call Ollama LLM with the given prompt.
        
        Args:
            prompt: The user prompt/question
            system: System prompt/instruction
            stream: Whether to stream the response
            temperature: Override default temperature
            
        Returns:
            Generated response text
        """
        temp = temperature if temperature is not None else self.config.temperature
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "temperature": temp,
            "top_p": self.config.top_p,
            "keep_alive": "5m",
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.config.timeout,
                )
                response.raise_for_status()

                if stream:
                    result = ""
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            result += data.get("response", "")
                    return result.strip()

                data = response.json()
                return data.get("response", "").strip()

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.config.max_retries:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                break

        raise RuntimeError(f"Ollama API error: {str(last_error)}")


def _extract_json_block(text: str) -> Optional[dict]:
    """Extract the first valid JSON object found in a text response."""
    text = text.strip()
    candidates: list[str] = [text]

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.find("```", start)
        if end != -1:
            candidates.append(text[start:end].strip())

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


class OllamaEvaluator:
    """LLM-based evaluator using Ollama"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        
    def evaluate(
        self,
        example: QAExample,
        answer: str,
    ) -> tuple[JudgeResult, int]:
        """
        Evaluate the answer using LLM.
        
        Args:
            example: QA example with context
            answer: The answer to evaluate
            
        Returns:
            Tuple of (JudgeResult, latency_ms)
        """
        start_time = time.time()
        
        # Prepare context string
        context_str = "\n".join(f"[{chunk.title}]: {chunk.text[:300]}" for chunk in example.context[:3])
        
        # Create evaluation prompt
        eval_prompt = f"""Question: {example.question}
Context:
{context_str}

Gold: {example.gold_answer}
Answer: {answer}

Is the answer correct? Return strict JSON only:
{{"score": 0 or 1, "reason": "short explanation"}}
"""
        
        try:
            response = self.llm.call(
                prompt=eval_prompt,
                system=EVALUATOR_SYSTEM,
                temperature=0.1,  # Low temperature for consistent evaluation
            )
            
            # Parse JSON response
            # Try to extract JSON from response
            result_data = _extract_json_block(response)
            if result_data is not None:
                score = 1 if result_data.get("score", 0) == 1 else 0
                reason = result_data.get("reason", "Evaluation completed")
            else:
                # Fallback: use simple normalization comparison
                score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
                reason = "Answer matches gold answer." if score == 1 else "Answer does not match gold answer."
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return JudgeResult(score=score, reason=reason), latency_ms
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            # Fallback evaluation
            score = 1 if normalize_answer(example.gold_answer) == normalize_answer(answer) else 0
            reason = f"Evaluation error, fallback comparison: {str(e)}"
            return JudgeResult(score=score, reason=reason), latency_ms


class OllamaReflector:
    """LLM-based reflector using Ollama"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        
    def reflect(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, int]:
        """
        Generate reflection on why answer failed and strategy for next attempt.
        
        Args:
            example: QA example with context
            attempt_id: Current attempt number
            answer: The incorrect answer
            judge: Evaluation result
            
        Returns:
            Tuple of (ReflectionEntry, latency_ms)
        """
        start_time = time.time()
        
        # Prepare context string
        context_str = "\n".join(
            f"[{chunk.title}]: {chunk.text[:300]}"
            for chunk in example.context[:3]
        )
        
        # Create reflection prompt
        reflect_prompt = f"""Question: {example.question}

Context:
{context_str}

Actor's Answer (Incorrect): {answer}

Gold Answer: {example.gold_answer}

Evaluation Feedback: {judge.reason}

Based on the failure, provide:
1. Why the answer was wrong
2. What strategy should be used in the next attempt to get the correct answer"""
        
        try:
            response = self.llm.call(
                prompt=reflect_prompt,
                system=REFLECTOR_SYSTEM,
                temperature=0.3,  # Moderate temperature for reflective thinking
            )
            
            parsed = _extract_json_block(response)
            if parsed:
                failure_reason = str(parsed.get("failure_reason", judge.reason))
                lesson = str(parsed.get("lesson", "Use the feedback to avoid repeating the same mistake."))
                next_strategy = str(parsed.get("next_strategy", "Extract evidence directly from context and answer concisely."))
            else:
                lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
                failure_reason = lines[0] if lines else judge.reason
                lesson = lines[1] if len(lines) > 1 else "Based on the feedback, improve evidence-grounded reasoning."
                next_strategy = response if len(response) > 50 else "Revise answer based on explicit evidence."
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=failure_reason,
                lesson=lesson,
                next_strategy=next_strategy,
            )
            
            return reflection, latency_ms
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson=f"Error in reflection: {str(e)}",
                next_strategy="Try a different approach based on the context.",
            )
            return reflection, latency_ms


class OllamaActor:
    """LLM-based actor using Ollama for ReAct reasoning"""
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
        
    def act(
        self,
        example: QAExample,
        reflection_memory: Optional[list[str]] = None,
    ) -> tuple[str, int]:
        """
        Generate answer using ReAct-style reasoning with LLM.
        
        Args:
            example: QA example with context
            reflection_memory: Previous reflections to inform next attempt
            
        Returns:
            Tuple of (answer, latency_ms)
        """
        start_time = time.time()
        
        # Prepare context string
        context_str = "\n".join(
            f"[{chunk.title}]: {chunk.text[:300]}"
            for chunk in example.context[:3]
        )
        
        # Build prompt with reflection memory if available
        memory_str = ""
        if reflection_memory and len(reflection_memory) > 0:
            memory_str = "\n\nPrevious Attempts and Lessons:\n"
            for mem in reflection_memory:
                memory_str += f"- {mem}\n"

        # Create ReAct-style prompt
        react_prompt = f"""Question: {example.question}

Context:
{context_str}
{memory_str}

Answer the question based on the context.
Give only the final answer in this forrmat:
FINAL ANSWER: [answer here]"""
        
        try:
            response = self.llm.call(
                prompt=react_prompt,
                system=ACTOR_SYSTEM,
                temperature=self.llm.config.temperature,
            )
            
            # Extract final answer from response
            answer = self._extract_answer(response)
            latency_ms = int((time.time() - start_time) * 1000)
            
            return answer, latency_ms
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return f"Error generating answer: {str(e)}", latency_ms
    
    def _extract_answer(self, response: str) -> str:
        """Extract final answer from LLM response"""
        # Look for "FINAL ANSWER:" pattern
        if "FINAL ANSWER:" in response:
            parts = response.split("FINAL ANSWER:")
            if len(parts) > 1:
                answer = parts[-1].strip().split("\n")[0].strip()
                return answer
        
        # Fallback: return last non-empty line
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[-1] if lines else "No answer generated"


class OllamaRuntime:
    """Main runtime orchestrator for ReAct and Reflexion agents with Ollama"""
    
    def __init__(self, config: Optional[OllamaRuntimeConfig] = None):
        self.config = config or OllamaRuntimeConfig()
        self.llm = OllamaLLM(self.config)
        self.actor = OllamaActor(self.llm)
        self.evaluator = OllamaEvaluator(self.llm)
        self.reflector = OllamaReflector(self.llm)
        
    def run_example(
        self,
        example: QAExample,
        agent_type: str = "react",
        max_attempts: int = 1,
    ) -> RunRecord:
        """
        Run ReAct/Reflexion agent on a QA example.
        
        Args:
            example: QA example to solve
            agent_type: "react" or "reflexion"
            max_attempts: Maximum attempts allowed
            
        Returns:
            RunRecord with complete execution trace
        """
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        
        final_answer = ""
        final_score = 0
        total_token_estimate = 0
        
        for attempt_id in range(1, max_attempts + 1):
            # Step 1: Actor generates answer (ReAct reasoning)
            answer, actor_latency = self.actor.act(example, reflection_memory)
            total_token_estimate += self._estimate_tokens(answer)
            
            # Step 2: Evaluator scores the answer
            judge, eval_latency = self.evaluator.evaluate(example, answer)
            total_token_estimate += 150  # Estimation for evaluator tokens
            
            total_latency = actor_latency + eval_latency
            
            # Create trace entry
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=self._estimate_tokens(answer),
                latency_ms=total_latency,
            )
            
            final_answer = answer
            final_score = judge.score
            
            # Step 3: If correct, stop
            if judge.score == 1:
                traces.append(trace)
                break
            
            # Step 4: If incorrect and have attempts left, run reflector (2nd hop)
            if agent_type == "reflexion" and attempt_id < max_attempts:
                reflection, reflector_latency = self.reflector.reflect(
                    example, attempt_id, answer, judge
                )
                reflections.append(reflection)
                reflection_memory.append(
                    f"Attempt {attempt_id}: {reflection.failure_reason} "
                    f"Lesson: {reflection.lesson} Strategy: {reflection.next_strategy}"
                )
                total_token_estimate += self._estimate_tokens(reflection.next_strategy)
                trace.reflection = reflection
                trace.latency_ms += reflector_latency
            
            traces.append(trace)
        
        # Determine failure mode
        failure_mode = (
            "none"
            if final_score == 1
            else self._infer_failure_mode(example, final_answer, traces, reflections)
        )
        
        total_latency = sum(t.latency_ms for t in traces)
        
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_token_estimate,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough estimate of tokens (approximately 4 chars per token)"""
        return max(1, len(text) // 4)
    
    @staticmethod
    def _infer_failure_mode(
        example: QAExample,
        answer: str,
        traces: Optional[list[AttemptTrace]] = None,
        reflections: Optional[list[ReflectionEntry]] = None,
    ) -> str:
        """Infer failure mode from answer and context"""
        if not answer:
            return "incomplete_multi_hop"
        if normalize_answer(answer) == normalize_answer(example.gold_answer):
            return "none"

        if traces and len(traces) >= 2:
            normalized_answers = [normalize_answer(t.answer) for t in traces if t.answer]
            if len(normalized_answers) >= 2 and len(set(normalized_answers)) == 1:
                return "looping"

        if reflections and len(reflections) >= 2:
            return "reflection_overfit"

        # Check if answer is too short (incomplete multi-hop)
        if len(answer.split()) < 2:
            return "incomplete_multi_hop"

        context_text = " ".join(chunk.text.lower() for chunk in example.context)
        answer_tokens = [tok.strip(".,;:!?()[]{}\"'").lower() for tok in answer.split()]
        out_of_context = [
            tok for tok in answer_tokens if tok and len(tok) > 4 and tok not in context_text
        ]
        if len(out_of_context) >= max(1, len(answer_tokens) // 3):
            return "entity_drift"

        # Default to wrong_final_answer
        return "wrong_final_answer"


# Convenience functions for compatibility with existing code

def create_runtime(
    base_url: str = "http://localhost:11434",
    model="phi3:latest",
    temperature: float = 0.7,
    timeout: int = 300,
    max_retries: int = 2,
) -> OllamaRuntime:
    """
    Factory function to create OllamaRuntime with custom config.
    
    Args:
        base_url: Ollama server URL
        model: Model name to use
        temperature: Sampling temperature
        timeout: Request timeout in seconds
        max_retries: Number of retry attempts for transient request failures
        
    Returns:
        Configured OllamaRuntime instance
    """
    config = OllamaRuntimeConfig(
        base_url=base_url,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    return OllamaRuntime(config)
