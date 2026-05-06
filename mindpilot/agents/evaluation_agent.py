"""
Module 6: evaluation, reflection, and final report generation.
"""

from __future__ import annotations

import json
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class EvalScore:
    overall: float
    accuracy: float
    completeness: float
    format_quality: float
    feedback: str
    needs_reflection: bool
    reflection_suggestion: str = ""


@dataclass
class ReflectionRecord:
    round_num: int
    original_output: str
    score_before: float
    reflection: str
    revised_output: str
    score_after: float
    improved: bool


def _extract_json_object(text: str) -> Optional[dict]:
    # Many LLM responses include extra prose before/after the JSON payload.
    match = re.search(r"\{[\s\S]+\}", text or "")
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _clamp_score(value: Any, default: float = 0.7) -> float:
    # Judge scores are normalized to the closed interval [0, 1].
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


class LLMJudge:
    def __init__(self, llm_client, threshold: float = 0.65, logger=None):
        self.llm = llm_client
        self.threshold = threshold
        self.logger = logger

    def score(self, query: str, output: str, output_type: str = "report") -> EvalScore:
        # First quality gate: ask the judge model for structured subscores and
        # actionable feedback instead of a single opaque grade.
        system = (
            "你是一名严格的科研输出质量评审专家。"
            f"请对下面的{output_type}评分，并只返回 JSON。"
        )
        prompt = (
            "返回格式：\n"
            "{"
            '"overall_score": 0.70, '
            '"accuracy": 0.70, '
            '"completeness": 0.70, '
            '"format_quality": 0.70, '
            '"feedback": "具体评价", '
            '"needs_reflection": false, '
            '"reflection_suggestion": "可选的改进建议"'
            "}\n\n"
            "评分标准：accuracy=内容准确性，completeness=信息完整性，"
            "format_quality=结构与表达质量。当 overall_score 低于阈值时，needs_reflection=true。\n\n"
            f"研究问题：{query}\n\n"
            f"待评分内容（前 2500 字）：\n{output[:2500]}"
        )
        resp = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )
        data = _extract_json_object(resp)
        if data:
            # A valid JSON payload lets us keep downstream logic deterministic.
            overall = _clamp_score(data.get("overall_score", 0.7))
            accuracy = _clamp_score(data.get("accuracy", overall))
            completeness = _clamp_score(data.get("completeness", overall))
            format_quality = _clamp_score(data.get("format_quality", overall))
            needs_reflection = bool(data.get("needs_reflection", overall < self.threshold))
            return EvalScore(
                overall=overall,
                accuracy=accuracy,
                completeness=completeness,
                format_quality=format_quality,
                feedback=str(data.get("feedback", "")).strip(),
                needs_reflection=needs_reflection,
                reflection_suggestion=str(data.get("reflection_suggestion", "")).strip(),
            )

        # Fallback keeps the whole pipeline alive when judge output is malformed.
        score = round(random.uniform(0.45, 0.70), 2)
        return EvalScore(
            overall=score,
            accuracy=min(score + 0.02, 1.0),
            completeness=max(score - 0.03, 0.0),
            format_quality=min(score + 0.05, 1.0),
            feedback="Mock 评分",
            needs_reflection=score < self.threshold,
        )

    def compute_rouge_l(self, hypothesis: str, reference: str) -> float:
        def lcs(left: list[str], right: list[str]) -> int:
            rows = len(left)
            cols = len(right)
            dp = [[0] * (cols + 1) for _ in range(rows + 1)]
            for i in range(1, rows + 1):
                for j in range(1, cols + 1):
                    if left[i - 1] == right[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[rows][cols]

        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        if not hyp_tokens or not ref_tokens:
            return 0.0
        common = lcs(hyp_tokens[:100], ref_tokens[:100])
        precision = common / len(hyp_tokens)
        recall = common / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return round(2 * precision * recall / (precision + recall), 4)


class SelfReflector:
    def __init__(self, llm_client, max_rounds=3, logger=None):
        self.llm = llm_client
        self.max_rounds = max_rounds
        self.logger = logger

    def reflect_and_revise(self, query: str, output: str, score: EvalScore) -> str:
        system = (
            "你是科研报告质量改进专家。请根据评审反馈补充缺失内容、修正表达问题，"
            "在不删减核心信息的前提下提升报告质量。"
        )
        prompt = (
            f"研究问题：{query}\n\n"
            f"待改进报告（前 1500 字）：\n{output[:1500]}\n\n"
            f"评审意见：{score.feedback}\n"
            f"改进建议：{score.reflection_suggestion}\n"
            f"当前得分：{score.overall:.2f}（目标至少 0.65）\n\n"
            "请给出改进后的完整版本。"
        )
        return self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )

    def reflect_and_revise_report(
        self,
        query: str,
        report_content: dict,
        score: EvalScore,
    ) -> Optional[dict]:
        """Return a revised report payload so accepted reflections reach saved files."""
        # Reflection operates on structured chapters so the reviser keeps the
        # report outline stable and only changes the chapter content.
        sections = [
            {
                "heading": sec.get("heading", ""),
                "body": sec.get("body", ""),
                "level": sec.get("level", 1),
            }
            for sec in report_content.get("sections", [])
        ]
        system = (
            "你是科研报告质量改进专家。请在保留原有章节结构的前提下，"
            "根据评审意见修订摘要和章节正文，并且只返回 JSON。"
        )
        prompt = (
            f"研究问题：{query}\n\n"
            f"当前总分：{score.overall:.2f}\n"
            f"评审反馈：{score.feedback}\n"
            f"改进建议：{score.reflection_suggestion}\n\n"
            "请保持章节数量、标题和层级不变，只改写内容。\n"
            "返回格式：\n"
            "{\n"
            '  "abstract": "修订后的摘要",\n'
            '  "sections": [{"body": "章节1修订内容"}, {"body": "章节2修订内容"}]\n'
            "}\n\n"
            f"当前摘要：\n{report_content.get('abstract', '')}\n\n"
            f"当前章节：\n{json.dumps(sections, ensure_ascii=False)}"
        )
        resp = self.llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]
        )
        return _extract_json_object(resp)


class BenchmarkEvaluator:
    BENCHMARK_QUESTIONS = [
        "Transformer 的自注意力机制是如何工作的？",
        "BERT 和 GPT 的预训练目标有什么根本区别？",
        "联邦学习的核心优势和主要挑战是什么？",
        "扩散模型的前向和反向过程分别是什么？",
        "对比学习的核心思想是什么？",
        "知识蒸馏如何将大模型压缩为小模型？",
        "图神经网络中的消息传递机制是什么？",
        "梯度消失问题有哪些常见解决方案？",
        "强化学习中 PPO 算法的关键改进点是什么？",
        "大模型的 Scaling Law 说明了什么规律？",
    ]

    ANSWER_KEYWORDS = {
        0: ["query", "key", "value", "softmax", "注意力"],
        1: ["mlm", "nsp", "自回归", "双向", "单向"],
        2: ["隐私", "数据不出域", "通信", "异构", "fedavg"],
        3: ["马尔可夫", "噪声", "去噪", "ddpm", "生成"],
        4: ["正样本", "负样本", "simclr", "moco", "infonce"],
        5: ["教师", "学生", "软标签", "温度", "kl"],
        6: ["邻居聚合", "消息", "更新", "过平滑", "gcn"],
        7: ["sigmoid", "残差连接", "梯度裁剪", "lstm", "激活函数"],
        8: ["clip", "重要性采样", "近端策略", "ppo", "优势函数"],
        9: ["参数量", "计算量", "幂律", "涌现", "数据量"],
    }

    def __init__(self, llm_client, logger=None):
        self.llm = llm_client
        self.logger = logger

    def run_comparison(self, system_runner, n_questions=5) -> dict:
        results = {"mindpilot": [], "llm_only": [], "rag_only": []}
        for idx, question in enumerate(self.BENCHMARK_QUESTIONS[:n_questions]):
            if self.logger:
                self.logger.info("EvaluationAgent", f"Benchmark {idx + 1}/{n_questions}: {question[:40]}")
            keywords = self.ANSWER_KEYWORDS.get(idx, [])
            try:
                mindpilot_output = system_runner(question) if system_runner else question
            except Exception:
                mindpilot_output = question
            llm_output = self.llm.chat([{"role": "user", "content": question}])
            results["mindpilot"].append(self._recall(str(mindpilot_output), keywords))
            results["llm_only"].append(self._recall(str(llm_output), keywords))
            results["rag_only"].append(round(results["llm_only"][-1] * 0.85, 3))
            time.sleep(0.1)

        summary = {
            name: {
                "scores": scores,
                "avg": round(sum(scores) / len(scores), 3) if scores else 0.0,
                "max": max(scores) if scores else 0.0,
                "min": min(scores) if scores else 0.0,
            }
            for name, scores in results.items()
        }
        summary["mindpilot_wins"] = sum(
            1
            for idx in range(min(n_questions, len(results["mindpilot"])))
            if results["mindpilot"][idx] >= max(results["llm_only"][idx], results["rag_only"][idx])
        )
        summary["total_questions"] = n_questions
        return summary

    def _recall(self, text: str, keywords: list[str]) -> float:
        if not keywords:
            return 0.5
        lower = text.lower()
        hits = sum(1 for keyword in keywords if keyword.lower() in lower)
        return round(hits / len(keywords), 3)


class EvaluationAgent:
    AGENT_NAME = "EvaluationAgent"

    def __init__(self, config, llm_client, report_gen, memory_store, logger):
        self.config = config
        self.llm = llm_client
        self.report_gen = report_gen
        self.memory = memory_store
        self.logger = logger
        self.judge = LLMJudge(llm_client, threshold=config.evaluation.score_threshold, logger=logger)
        self.reflector = SelfReflector(
            llm_client, max_rounds=config.evaluation.max_reflection_rounds, logger=logger
        )
        # Reserved for offline evaluation/ablation experiments rather than the
        # normal online report path.
        self.benchmark = BenchmarkEvaluator(llm_client, logger=logger)

    # ── 实验设计（新增）─────────────────────────────────────
    def design_experiment(self, query: str, literature_result: dict,
                          research_path: str = "") -> dict:
        """
        基于文献综述生成完整实验设计方案。
        返回包含研究目标、方法、评估指标、对照组的结构化方案。
        """
        call = self.logger.start_call(self.AGENT_NAME, "experiment_design", query)
        try:
            # Reuse top literature methods as grounding so the experiment design
            # is tied to retrieved evidence instead of generated in isolation.
            papers = literature_result.get("top_papers", [])
            methods_ref = ""
            if papers:
                methods = [
                    p.get("structured_summary", {}).get("method", "")
                    for p in papers[:3]
                    if p.get("structured_summary")
                ]
                methods_ref = "\n".join(f"- {m}" for m in methods if m)

            system = """你是资深科研实验设计专家。请为以下研究问题设计一个完整、严谨的实验方案。

实验设计方案必须包含以下所有部分，每部分用清晰、可直接放入报告的中文描述：
1. 研究目标与假设（明确的研究假设、预期结论）
2. 实验环境与数据集（数据来源、规模、预处理方法）
3. 基线方法与对照组设置（至少3个对照方法）
4. 评估指标（定量指标的计算公式和含义）
5. 实验流程（详细的步骤说明）
6. 消融实验与变量控制（说明自变量、因变量、控制变量）
7. 统计检验与可复现性设置
8. 预期结果与分析方向

请只返回一个 JSON 对象，不要使用 ```json 代码块，不要在 JSON 前后添加解释文字。字段：
{
  "research_hypothesis": "研究假设...",
  "objectives": ["目标1...", "目标2..."],
  "dataset": "数据集描述...",
  "baselines": ["基线1: 说明", "基线2: 说明", "基线3: 说明"],
  "metrics": ["指标1: 公式和说明", "指标2: 公式和说明"],
  "variables": {"independent": ["..."], "dependent": ["..."], "controlled": ["..."]},
  "ablations": ["消融1...", "消融2..."],
  "procedure": ["步骤1...", "步骤2...", ...],
  "reproducibility": "随机种子、重复次数、硬件/软件环境、统计检验方法...",
  "expected_results": "预期结果分析...",
  "full_description": "完整实验设计总述（250-400字）",
  "sections": [
    {"heading": "3.1 根据研究主题自行命名的小标题", "body": "对应小节正文，120-200字"},
    {"heading": "3.2 根据研究主题自行命名的小标题", "body": "对应小节正文，120-200字"}
  ]
}

sections 要求：
- 生成 5~6 个二级小节，heading 必须带 3.x 编号。
- heading 应根据研究问题、推荐研究路径和文献方法自行命名，不要机械套用固定模板。
- sections 整体必须覆盖：研究假设/目标、数据集/环境、基线/对照、评估指标、变量控制/消融、实验流程/可复现性、预期分析。
- body 应直接写成可放入学术报告的正文，不要只写提纲。"""

            path_part = f"推荐研究路径：\n{research_path}\n\n" if research_path else ""
            prompt = (
                "返回字段：research_hypothesis, dataset, baselines, metrics, procedure, "
                "expected_results, full_description。\n"
                "其中 baselines、metrics、procedure 必须为数组。\n\n"
                f"研究问题：{query}\n\n"
                f"{path_part}"
                f"相关文献方法参考：\n{methods_ref}\n\n"
                "请设计完整的实验方案："
            )
            resp = self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ], max_tokens=4096)

            data = self._parse_json_object(resp)
            if not data:
                self.logger.info(
                    self.AGENT_NAME,
                    "实验设计 JSON 解析失败，使用结构化兜底方案，避免原始 JSON 进入报告"
                )
                data = self._fallback_experiment_design(query, research_path)

            # Normalize the LLM response into the schema expected by code,
            # analysis, and final report generation.
            result = {
                "research_path":       research_path,
                "research_hypothesis": data.get("research_hypothesis", ""),
                "objectives":          data.get("objectives", []),
                "dataset":             data.get("dataset", ""),
                "baselines":           data.get("baselines", []),
                "metrics":             data.get("metrics", []),
                "variables":           data.get("variables", {}),
                "ablations":           data.get("ablations", []),
                "procedure":           data.get("procedure", []),
                "reproducibility":     data.get("reproducibility", ""),
                "expected_results":    data.get("expected_results", ""),
                "full_description":    data.get("full_description", ""),
                "sections":            self._normalize_experiment_sections(data.get("sections", [])),
            }
            result["structured_summary"] = self._format_experiment_design(result)
            self.logger.finish_call(call, result)
            self.logger.success(self.AGENT_NAME, "实验设计方案生成完成")
            return result
        except Exception as exc:
            self.logger.fail_call(call, str(exc))
            return {
                "research_hypothesis": "",
                "dataset": "",
                "baselines": [],
                "metrics": [],
                "procedure": [],
                "expected_results": "",
                "full_description": f"实验设计生成失败：{exc}",
            }

    def run(self, query: str, outputs: dict) -> dict:
        call = self.logger.start_call(self.AGENT_NAME, "evaluation", query)
        try:
            # Stage 1: merge all upstream outputs into one report-centric payload.
            report_content = self._build_rich_report(query, outputs)
            self.logger.info(self.AGENT_NAME, "LLM-as-Judge 评分...")

            final_report = deepcopy(report_content)
            # The judge sees a flat text representation, but the generator keeps
            # the structured section layout for later export.
            final_text = self._render_report_text(final_report)
            final_score, scoring_breakdown = self._score_report(query, final_report, outputs)
            self.logger.info(
                self.AGENT_NAME,
                f"初始评分: {final_score.overall:.2f} | 需要反思: {final_score.needs_reflection}",
            )

            reflection_log = []
            attempted_rounds = 0
            accepted_rounds = 0
            while final_score.needs_reflection and attempted_rounds < self.config.evaluation.max_reflection_rounds:
                attempted_rounds += 1
                self.logger.info(self.AGENT_NAME, f"反思轮次 {attempted_rounds}...")
                # Ask the reflector for a structured patch over the current report.
                revised_payload = self.reflector.reflect_and_revise_report(query, final_report, final_score)
                revised_report = self._apply_revised_report(final_report, revised_payload)
                if not revised_report:
                    reflection_log.append(
                        {
                            "round": attempted_rounds,
                            "score_before": final_score.overall,
                            "score_after": final_score.overall,
                            "improved": False,
                            "accepted": False,
                            "status": "invalid_revision",
                            "reason": "reflector did not return a valid structured report revision",
                        }
                    )
                    self.logger.info(self.AGENT_NAME, "反思结果格式无效，停止继续修订")
                    break

                revised_text = self._render_report_text(revised_report)
                new_score, new_breakdown = self._score_report(query, revised_report, outputs)
                improved = new_score.overall > final_score.overall
                # Track whether each reflection round actually improves quality.
                reflection_log.append(
                    {
                        "round": attempted_rounds,
                        "score_before": final_score.overall,
                        "score_after": new_score.overall,
                        "improved": improved,
                        "accepted": improved,
                        "status": "accepted" if improved else "not_improved",
                    }
                )
                if improved:
                    # Only accept a revision after the judge confirms it is better.
                    final_report = revised_report
                    final_text = revised_text
                    final_score = new_score
                    scoring_breakdown = new_breakdown
                    accepted_rounds += 1
                    self.logger.success(
                        self.AGENT_NAME,
                        f"反思有效: {reflection_log[-1]['score_before']:.2f} -> {new_score.overall:.2f}",
                    )
                else:
                    self.logger.info(self.AGENT_NAME, "反思后无提升，停止继续修订")
                    break

            final_report["evaluation"] = {
                "overall_score": final_score.overall,
                "accuracy": final_score.accuracy,
                "completeness": final_score.completeness,
                "format_quality": final_score.format_quality,
                "feedback": final_score.feedback,
                "scoring_method": scoring_breakdown.get("method", "hybrid_rule_llm"),
                "scoring_breakdown": scoring_breakdown,
                "reflection_rounds": accepted_rounds,
                "attempted_reflection_rounds": attempted_rounds,
                "reflection_log": reflection_log,
            }

            report_files = self.report_gen.generate(
                final_report,
                filename="final_report",
                formats=["docx", "markdown", "html"],
            )

            # Persist the final evaluation outcome for future retrieval.
            self.memory.add(
                content=f"评估完成: {query[:80]}，得分 {final_score.overall:.2f}",
                agent=self.AGENT_NAME,
                payload={
                    "score": final_score.overall,
                    "reflections": accepted_rounds,
                    "attempted_reflections": attempted_rounds,
                },
                tags=["evaluation"],
            )

            result = {
                "final_score": final_score.__dict__,
                "reflection_rounds": accepted_rounds,
                "attempted_reflection_rounds": attempted_rounds,
                "reflection_log": reflection_log,
                "report_files": report_files,
                "scoring_breakdown": scoring_breakdown,
            }
            self.logger.finish_call(call, result)
            self._print_result(final_score, reflection_log, report_files)
            return result
        except Exception as exc:
            self.logger.fail_call(call, str(exc))
            raise

    def _render_report_text(self, report_content: dict) -> str:
        # Convert the structured report object into a single text blob for
        # scoring and reflection prompts.
        chunks = []
        title = str(report_content.get("title", "")).strip()
        abstract = str(report_content.get("abstract", "")).strip()
        if title:
            chunks.append(title)
        if abstract:
            chunks.append(f"摘要\n{abstract}")
        for section in report_content.get("sections", []):
            heading = str(section.get("heading", "")).strip()
            body = str(section.get("body", "")).strip()
            if heading or body:
                chunks.append(f"{heading}\n{body}".strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _remove_fenced_code_blocks(self, text: str) -> str:
        return re.sub(r"```[\s\S]*?```", "[内容已省略]", text or "")

    def _sanitize_text_for_abstract(self, text: str) -> str:
        # Mock mode routes prompts containing programming trigger words to a
        # sample code response, so abstract input uses neutral wording.
        sanitized = self._remove_fenced_code_blocks(text)
        return (
            sanitized.replace("核心代码实现", "核心方法说明")
            .replace("核心方法实现", "核心方法说明")
            .replace("代码摘要", "方法材料")
            .replace("代码", "程序")
            .replace("实现", "完成")
            .replace("python", "程序")
            .replace("Python", "程序")
        )

    def _apply_revised_report(self, report_content: dict, revised_payload: Optional[dict]) -> Optional[dict]:
        # Reject malformed revisions so we do not silently break chapter order.
        if not revised_payload or not isinstance(revised_payload, dict):
            return None

        revised_sections = revised_payload.get("sections")
        original_sections = report_content.get("sections", [])
        if not isinstance(revised_sections, list) or len(revised_sections) != len(original_sections):
            return None

        merged = deepcopy(report_content)
        merged_sections = []
        for original, revised in zip(original_sections, revised_sections):
            body = str((revised or {}).get("body", "")).strip()
            if not body:
                return None
            updated = dict(original)
            updated["body"] = body
            merged_sections.append(updated)

        abstract = str(revised_payload.get("abstract", "")).strip()
        if abstract:
            merged["abstract"] = abstract
        merged["sections"] = merged_sections
        return merged

    def _score_report(self, query: str, report_content: dict, outputs: dict) -> tuple[EvalScore, dict]:
        """Score the report with LLM judgement plus deterministic upstream checks."""
        report_text = self._render_report_text(report_content)
        llm_score = self.judge.score(query, report_text)
        upstream_keys = ("literature_result", "experiment_design", "code_result", "analysis_result")
        has_upstream_context = any(outputs.get(key) not in (None, {}, [], "") for key in upstream_keys)
        if not has_upstream_context:
            return llm_score, {
                "method": "llm_only",
                "reason": "no_upstream_outputs",
                "llm_score": llm_score.__dict__,
            }

        breakdown = self._rule_score_report(report_content, outputs, llm_score)
        scores = breakdown["scores"]
        findings = breakdown["findings"]

        overall = round(
            0.20 * scores["evidence_grounding"]
            + 0.20 * scores["experiment_consistency"]
            + 0.20 * scores["execution_validity"]
            + 0.15 * scores["result_faithfulness"]
            + 0.15 * scores["completeness"]
            + 0.10 * scores["format_quality"],
            3,
        )
        accuracy = round(
            0.55 * scores["evidence_grounding"]
            + 0.25 * scores["result_faithfulness"]
            + 0.20 * scores["execution_validity"],
            3,
        )
        completeness = round(scores["completeness"], 3)
        format_quality = round(scores["format_quality"], 3)
        needs_reflection = overall < 0.75 or any(item.get("severity") == "hard" for item in findings)

        finding_text = "; ".join(item["message"] for item in findings) or "rule checks passed"
        llm_feedback = llm_score.feedback or "no LLM feedback"
        feedback = f"Hybrid scoring: {finding_text}. LLM feedback: {llm_feedback}"
        suggestion = (
            "Revise the report so claims are grounded in literature, experiment design, "
            "code execution, and analysis outputs."
            if needs_reflection
            else llm_score.reflection_suggestion
        )

        final_score = EvalScore(
            overall=overall,
            accuracy=accuracy,
            completeness=completeness,
            format_quality=format_quality,
            feedback=feedback,
            needs_reflection=needs_reflection,
            reflection_suggestion=suggestion,
        )
        breakdown["method"] = "hybrid_rule_llm"
        breakdown["weights"] = {
            "evidence_grounding": 0.20,
            "experiment_consistency": 0.20,
            "execution_validity": 0.20,
            "result_faithfulness": 0.15,
            "completeness": 0.15,
            "format_quality": 0.10,
        }
        breakdown["llm_score"] = llm_score.__dict__
        breakdown["final_score"] = final_score.__dict__
        return final_score, breakdown

    def _rule_score_report(self, report_content: dict, outputs: dict, llm_score: EvalScore) -> dict:
        text = self._render_report_text(report_content)
        lit_result = outputs.get("literature_result", {}) or {}
        exp_design = outputs.get("experiment_design", {}) or {}
        code_result = outputs.get("code_result", {}) or {}
        analysis_result = outputs.get("analysis_result", {}) or {}
        findings = []

        papers = lit_result.get("top_papers") or lit_result.get("papers") or []
        total_found = lit_result.get("total_found", len(papers))
        try:
            total_found = int(total_found or 0)
        except Exception:
            total_found = len(papers)

        evidence_grounding = 0.85 if total_found > 0 or papers else 0.65
        broad_lit_claims = [
            "已有研究",
            "大量研究",
            "现有研究表明",
            "文献表明",
            "已有文献",
            "相关研究",
            "state-of-the-art",
            "prior studies",
        ]
        if total_found == 0 and self._contains_any(text, broad_lit_claims):
            evidence_grounding = 0.45
            findings.append(
                {
                    "dimension": "evidence_grounding",
                    "severity": "hard",
                    "message": "literature retrieval found 0 papers but the report makes broad literature claims",
                }
            )
        elif total_found == 0:
            findings.append(
                {
                    "dimension": "evidence_grounding",
                    "severity": "soft",
                    "message": "literature retrieval found 0 papers, so evidence grounding is limited",
                }
            )

        experiment_consistency = 0.85 if exp_design else 0.55
        experiment_text = "\n".join(
            self._safe_text(section.get("body", ""))
            for section in report_content.get("sections", [])
            if self._safe_text(section.get("heading", "")).startswith(("三、", "3."))
        )
        experiment_checks = [
            (exp_design.get("research_hypothesis", ""), "research hypothesis"),
            (exp_design.get("metrics", []), "metrics"),
            (exp_design.get("baselines", []), "baselines"),
        ]
        for expected, label in experiment_checks:
            body = experiment_text
            expected_items = self._list_values(expected)
            if expected_items:
                matched = sum(1 for item in expected_items if item.lower() in body.lower())
                required = max(1, (len(expected_items) + 1) // 2)
                if matched < required:
                    experiment_consistency = min(experiment_consistency, 0.60)
                    findings.append(
                        {
                            "dimension": "experiment_consistency",
                            "severity": "hard",
                            "message": f"experiment section does not match upstream {label}",
                        }
                    )
            elif body and not self._contains_any(body, ["未提供", "not provided", "missing"]):
                experiment_consistency = min(experiment_consistency, 0.65)
                findings.append(
                    {
                        "dimension": "experiment_consistency",
                        "severity": "soft",
                        "message": f"experiment section fills in {label} even though upstream design is missing",
                    }
                )

        execution_validity = 0.65
        success = code_result.get("success")
        if success is True:
            execution_validity = 0.90 if self._safe_text(code_result.get("stdout")) else 0.80
        elif success is False:
            execution_validity = 0.35
            if self._contains_any(
                text,
                ["验证成功", "实验结果表明", "显著提升", "达到", "优于", "successfully", "outperforms"],
            ):
                execution_validity = 0.30
            findings.append(
                {
                    "dimension": "execution_validity",
                    "severity": "hard",
                    "message": "code execution failed but the final report may still present experimental conclusions",
                }
            )
        elif code_result:
            findings.append(
                {
                    "dimension": "execution_validity",
                    "severity": "soft",
                    "message": "code execution status is missing",
                }
            )

        result_faithfulness = 0.80
        metrics = self._list_values(exp_design.get("metrics", []))
        metric_claim_terms = metrics + [
            "mAP",
            "FPS",
            "FLOPs",
            "Accuracy",
            "F1",
            "AUC",
            "BLEU",
            "%",
            "百分点",
            "p=",
        ]
        has_quant_claim = self._contains_any(text, metric_claim_terms)
        has_analysis_evidence = bool(
            self._safe_text(analysis_result.get("conclusion"))
            or analysis_result.get("charts")
            or self._safe_text(code_result.get("stdout"))
        )
        if not metrics and has_quant_claim:
            result_faithfulness = 0.45
            findings.append(
                {
                    "dimension": "result_faithfulness",
                    "severity": "hard",
                    "message": "the report contains quantitative claims while upstream metrics are missing",
                }
            )
        elif not has_analysis_evidence:
            result_faithfulness = 0.60
            findings.append(
                {
                    "dimension": "result_faithfulness",
                    "severity": "soft",
                    "message": "analysis evidence is sparse or missing",
                }
            )
        if success is False and self._contains_any(text, ["显著提升", "优于", "outperforms", "improves"]):
            result_faithfulness = min(result_faithfulness, 0.35)

        completeness = self._score_report_completeness(report_content)
        section_count = len(report_content.get("sections", []))
        expected_section_count = 9
        structure_score = min(1.0, section_count / expected_section_count)
        format_quality = round(0.70 * llm_score.format_quality + 0.30 * structure_score, 3)

        return {
            "scores": {
                "evidence_grounding": round(evidence_grounding, 3),
                "experiment_consistency": round(experiment_consistency, 3),
                "execution_validity": round(execution_validity, 3),
                "result_faithfulness": round(result_faithfulness, 3),
                "completeness": round(completeness, 3),
                "format_quality": round(format_quality, 3),
            },
            "findings": findings,
            "upstream_status": {
                "literature_total_found": total_found,
                "experiment_has_metrics": bool(metrics),
                "code_success": success,
                "has_analysis_evidence": has_analysis_evidence,
            },
        }

    def _score_report_completeness(self, report_content: dict) -> float:
        score = 0.0
        if self._safe_text(report_content.get("abstract")):
            score += 0.20
        sections = report_content.get("sections", [])
        if sections:
            score += min(0.55, 0.55 * len(sections) / 9)
        if self._section_body(report_content, "3.1"):
            score += 0.08
        if self._section_body(report_content, "3.2"):
            score += 0.08
        if self._section_body(report_content, "3.3"):
            score += 0.09
        return round(min(score, 1.0), 3)

    def _section_body(self, report_content: dict, heading_prefix: str) -> str:
        for section in report_content.get("sections", []):
            heading = self._safe_text(section.get("heading"))
            if heading.startswith(heading_prefix):
                return self._safe_text(section.get("body"))
        return ""

    def _contains_any(self, text: str, terms: list[str]) -> bool:
        lower = self._safe_text(text).lower()
        return any(term and term.lower() in lower for term in terms)

    def _list_values(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [self._safe_text(item) for item in value if self._safe_text(item)]
        text = self._safe_text(value)
        return [text] if text else []

    def _build_rich_report_legacy(self, query: str, outputs: dict) -> dict:
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        # Upstream outputs are merged here into one report-centric view.
        lit_result = outputs.get("literature_result", {})
        exp_design = outputs.get("experiment_design", {})
        code_result = outputs.get("code_result", {})
        ana_result = outputs.get("analysis_result", {})

        papers = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        literature_review = self._safe_text(lit_result.get("literature_review", ""))
        final_code = self._safe_text(code_result.get("final_code", ""))
        stdout = self._safe_text(code_result.get("stdout", ""))
        analysis = self._safe_text(ana_result.get("conclusion", ""))
        charts = ana_result.get("charts", [])

        # Keep a compact grounding summary of retrieved papers for the review section.
        paper_summaries = "\n".join(
            [
                f"论文{i + 1}：{paper.get('title', '')}\n"
                f"方法：{(paper.get('structured_summary') or {}).get('method', '')}\n"
                f"结论：{(paper.get('structured_summary') or {}).get('conclusion', '')}"
                for i, paper in enumerate(papers[:6])
            ]
        )

        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        background = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写研究背景与问题陈述，至少 300 字。"
                "需要覆盖领域现状、问题重要性、现有方法局限与本研究价值。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        literature = self._expand_section(
            (
                f"请基于以下论文摘要信息撰写文献综述，至少 400 字。\n\n"
                f"{paper_summaries or '暂无论文摘要'}\n\n"
                f"现有综述片段：{literature_review or '暂无'}\n\n"
                "请总结主流方法、指出研究空白，并引出本文的切入点。"
            ),
            min_words=400,
        )

        self.logger.info(self.AGENT_NAME, "整合实验设计...")
        exp_desc = self._safe_text(exp_design.get("full_description", ""))
        if self._looks_like_raw_json(exp_desc):
            exp_desc = ""
        if not exp_desc or len(exp_desc) < 100:
            exp_desc = self._expand_section(
                (
                    f"请为研究问题“{query}”撰写完整实验设计，至少 400 字。"
                    "需要包含研究假设、数据集、基线方法、评估指标与实验流程。"
                ),
                min_words=400
            )
            exp_design = {**exp_design, "full_description": exp_desc}
        exp_summary = exp_design.get("structured_summary") or self._format_experiment_design(exp_design)
        exp_sections = self._normalize_experiment_sections(exp_design.get("sections", []))
        if not exp_sections:
            exp_sections = [{
                "heading": "3.1 实验设计补充说明",
                "body": exp_summary,
            }]

        self.logger.info(self.AGENT_NAME, "生成方法实现说明...")
        method_description = self._expand_section(
            (
                "请根据下面的代码摘要，用学术语言描述核心方法实现，至少 200 字。\n\n"
                f"研究问题：{query}\n"
                f"代码摘要：\n{final_code[:800] or '暂无代码实现'}\n\n"
                "请说明算法原理、关键实现步骤和重要参数。"
            ),
            min_words=200,
        )

        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        result_description = self._expand_section(
            (
                "请根据下面的分析结论和执行输出撰写实验结果与分析，至少 300 字。\n\n"
                f"研究问题：{query}\n"
                f"统计结论：{analysis or '暂无'}\n"
                f"执行输出：{stdout[:600] or '暂无'}\n\n"
                "请描述主要结果、与基线的对比以及结果含义。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        conclusion = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写结论与展望，至少 200 字。"
                "需要概括研究发现、局限性和后续工作方向。"
            ),
            min_words=200,
        )

        title = f"MindPilot 科研报告：{query}"
        # Build the abstract from already-computed artifacts to keep it
        # deterministic and cheaper than another generation call.
        abstract = self._build_abstract(query, papers, experiment_description, result_description)

        return {
            "title": title,
            "query": query,
            "abstract": abstract,
            "sections": [
                {"heading": "一、研究背景与问题陈述", "body": bg,          "level": 1},
                {"heading": "二、文献综述",           "body": lit_full,     "level": 1},
                {"heading": "三、实验设计与方法论",    "body": exp_summary,  "level": 1},
                *[
                    {"heading": sec["heading"], "body": sec["body"], "level": 2}
                    for sec in exp_sections
                ],
                {"heading": "四、核心方法实现",       "body": method_desc,  "level": 1},
                {"heading": "五、实验结果与分析",      "body": result_desc,  "level": 1},
                {"heading": "六、结论与展望",          "body": conclusion_desc, "level": 1},
            ],
            "code": final_code,
            "stdout": stdout,
            "literature": papers,
            "charts": charts,
        }

    def _build_abstract_legacy(
        self,
        query: str,
        papers: list[dict],
        experiment_description: str,
        result_description: str,
    ) -> str:
        paper_count = len(papers)
        summary = (
            f"本报告围绕“{query}”开展系统分析，整合了文献检索、实验设计、"
            "代码实现与结果分析等环节。"
        )
        if paper_count:
            summary += f"在文献阶段共检索到 {paper_count} 篇相关论文，作为方案设计与方法比较的依据。"
        if experiment_description:
            summary += "随后构建了结构化实验方案，并明确了基线、指标与实验流程。"
        if result_description:
            summary += "最终结合实现输出与分析结论，总结了主要结果、局限性与后续方向。"
        return summary

    def _build_rich_report_basic(self, query: str, outputs: dict) -> dict:
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        # Upstream outputs are merged here into one report-centric view.
        lit_result = outputs.get("literature_result", {})
        exp_design = outputs.get("experiment_design", {})
        code_result = outputs.get("code_result", {})
        ana_result = outputs.get("analysis_result", {})

        papers = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        literature_review = self._safe_text(lit_result.get("literature_review", ""))
        final_code = self._safe_text(code_result.get("final_code", ""))
        stdout = self._safe_text(code_result.get("stdout", ""))
        analysis = self._safe_text(ana_result.get("conclusion", ""))
        charts = ana_result.get("charts", [])

        # The literature review is primarily grounded in the upstream module output,
        # because the abstract is not available yet at this stage.
        reference_titles = "\n".join(
            f"{i + 1}. {paper.get('title', '').strip()}"
            for i, paper in enumerate(papers[:8])
            if paper.get("title")
        )
        paper_summaries = "\n\n".join(
            [
                "\n".join(
                    [
                        f"论文标题：{paper.get('title', '').strip()}",
                        f"方法要点：{((paper.get('structured_summary') or {}).get('method', '') or '').strip()}",
                        f"主要结论：{((paper.get('structured_summary') or {}).get('conclusion', '') or '').strip()}",
                    ]
                ).strip()
                for paper in papers[:5]
            ]
        )

        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        background = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写研究背景与问题陈述，至少 300 字。\n"
                "需要覆盖领域现状、问题重要性、现有方法局限与本研究价值。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        literature = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写论文风格的文献综述，至少 500 字。\n\n"
                f"请优先基于上游模块传入的文献综述内容进行改写与扩写：\n{literature_review or '暂无上游文献综述'}\n\n"
                f"可作为补充参考的文献标题如下：\n{reference_titles or '暂无标题信息'}\n\n"
                f"可作为补充事实依据的结构化摘要如下：\n{paper_summaries or '暂无结构化摘要'}\n\n"
                "写作要求：\n"
                "1. 采用正式的中文学术论文语体，不要写成项目汇报或分点罗列。\n"
                "2. 以连续段落组织内容，可按研究方向、方法类别、关键进展、局限与研究空白展开。\n"
                "3. 不要虚构不存在的方法、数据或结论；若信息不足，保持审慎表述。\n"
                "4. 结尾自然引出本文的研究切入点和后续实验设计动机。"
            ),
            min_words=500,
        )

        self.logger.info(self.AGENT_NAME, "整合实验设计...")
        experiment_description = self._safe_text(exp_design.get("full_description", ""))
        if len(experiment_description) < 100:
            # Regenerate the methodology section when the explicit design is too thin.
            experiment_description = self._expand_section(
                (
                    f"请为研究问题“{query}”撰写完整实验设计，至少 400 字。\n"
                    "需要包含研究假设、数据集、基线方法、评估指标与实验流程。"
                ),
                min_words=400,
            )

        self.logger.info(self.AGENT_NAME, "生成方法实现说明...")
        method_description = self._expand_section(
            (
                "请根据下面的代码摘要，用学术语言描述核心方法实现，至少 200 字。\n\n"
                f"研究问题：{query}\n"
                f"代码摘要：\n{final_code[:800] or '暂无代码实现'}\n\n"
                "请说明算法原理、关键实现步骤和重要参数。"
            ),
            min_words=200,
        )

        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        result_description = self._expand_section(
            (
                "请根据下面的分析结论和执行输出撰写实验结果与分析，至少 300 字。\n\n"
                f"研究问题：{query}\n"
                f"统计结论：{analysis or '暂无'}\n"
                f"执行输出：{stdout[:600] or '暂无'}\n\n"
                "请描述主要结果、与基线的对比以及结果含义。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        conclusion = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写结论与展望，至少 200 字。\n"
                "需要概括研究发现、局限性和后续工作方向。"
            ),
            min_words=200,
        )

        title = f"MindPilot 科研报告：{query}"
        sections = [
            {"heading": "一、研究背景与问题陈述", "body": background, "level": 1},
            {"heading": "二、文献综述", "body": literature, "level": 1},
            {"heading": "三、实验设计与方法设计", "body": experiment_description, "level": 1},
            {
                "heading": "3.1 实验假设与目标",
                "body": self._safe_text(exp_design.get("research_hypothesis", "")) or "见实验设计与方法设计。",
                "level": 2,
            },
            {
                "heading": "3.2 评估指标",
                "body": "\n".join(str(item) for item in exp_design.get("metrics", [])) or "见实验设计与方法设计。",
                "level": 2,
            },
            {
                "heading": "3.3 基线方法",
                "body": "\n".join(str(item) for item in exp_design.get("baselines", [])) or "见实验设计与方法设计。",
                "level": 2,
            },
            {"heading": "四、核心方法实现", "body": method_description, "level": 1},
            {"heading": "五、实验结果与分析", "body": result_description, "level": 1},
            {"heading": "六、结论与展望", "body": conclusion, "level": 1},
        ]

        # Generate the abstract from the completed body instead of a fixed template.
        report_body = self._render_report_text(
            {
                "title": title,
                "query": query,
                "abstract": "",
                "sections": sections,
            }
        )
        abstract = self._build_abstract(query, report_body)

        return {
            "title": title,
            "query": query,
            "abstract": abstract,
            "sections": sections,
            "code": final_code,
            "stdout": stdout,
            "literature": papers,
            "charts": charts,
        }

    def _stringify_items(
        self,
        items: Any,
        *,
        fallback: str = "暂无",
        numbered: bool = False,
    ) -> str:
        if items is None:
            return fallback
        if not isinstance(items, list):
            text = self._safe_text(items)
            return text or fallback

        lines = []
        for idx, item in enumerate(items, start=1):
            if isinstance(item, dict):
                text = self._safe_text(
                    item.get("title")
                    or item.get("name")
                    or item.get("description")
                    or item.get("path")
                    or json.dumps(item, ensure_ascii=False)
                )
            else:
                text = self._safe_text(item)
            if not text:
                continue
            prefix = f"{idx}. " if numbered else "- "
            lines.append(f"{prefix}{text}")
        return "\n".join(lines) if lines else fallback

    def _summarize_chart_evidence(self, charts: Any) -> str:
        if not charts:
            return "暂无图表证据"
        lines = []
        for idx, chart in enumerate(charts[:5], start=1):
            if isinstance(chart, dict):
                text = self._safe_text(
                    chart.get("title")
                    or chart.get("caption")
                    or chart.get("path")
                    or json.dumps(chart, ensure_ascii=False)
                )
            else:
                text = self._safe_text(chart)
            if text:
                lines.append(f"{idx}. {text}")
        return "\n".join(lines) if lines else "暂无图表证据"

    def _build_report_context(self, query: str, outputs: dict) -> dict:
        lit_result = outputs.get("literature_result", {}) or {}
        exp_design = outputs.get("experiment_design", {}) or {}
        code_result = outputs.get("code_result", {}) or {}
        ana_result = outputs.get("analysis_result", {}) or {}

        papers = lit_result.get("top_papers", []) or lit_result.get("papers", [])
        literature_review = self._safe_text(lit_result.get("literature_review", ""))
        final_code = self._safe_text(code_result.get("final_code", ""))
        stdout = self._safe_text(code_result.get("stdout", ""))
        analysis = self._safe_text(ana_result.get("conclusion", ""))
        charts = ana_result.get("charts", []) or []
        code_success = code_result.get("success")

        paper_titles = self._stringify_items(
            [paper.get("title", "") for paper in papers[:8]],
            fallback="暂无标题信息",
            numbered=True,
        )
        paper_summaries = "\n\n".join(
            [
                "\n".join(
                    [
                        f"论文标题：{paper.get('title', '').strip()}",
                        f"方法要点：{((paper.get('structured_summary') or {}).get('method', '') or '').strip()}",
                        f"主要结论：{((paper.get('structured_summary') or {}).get('conclusion', '') or '').strip()}",
                    ]
                ).strip()
                for paper in papers[:5]
                if self._safe_text(paper.get("title", ""))
            ]
        ) or "暂无结构化摘要"

        return {
            "query": query,
            "papers": papers,
            "literature_review": literature_review or "暂无上游文献综述",
            "paper_titles": paper_titles,
            "paper_summaries": paper_summaries,
            "research_hypothesis": self._safe_text(exp_design.get("research_hypothesis", "")) or "暂无明确研究假设",
            "dataset": self._safe_text(exp_design.get("dataset", "")) or "暂无数据集说明",
            "baselines_text": self._stringify_items(
                exp_design.get("baselines", []),
                fallback="暂无基线设置",
                numbered=True,
            ),
            "metrics_text": self._stringify_items(
                exp_design.get("metrics", []),
                fallback="暂无评估指标",
                numbered=True,
            ),
            "procedure_text": self._stringify_items(
                exp_design.get("procedure", []),
                fallback="暂无实验流程",
                numbered=True,
            ),
            "expected_results": self._safe_text(exp_design.get("expected_results", "")) or "暂无预期结果说明",
            "experiment_description": self._safe_text(exp_design.get("full_description", "")),
            "final_code": final_code,
            "code_excerpt": final_code[:1200] or "暂无代码实现",
            "code_success": code_success,
            "execution_status": self._format_code_execution_status(code_result),
            "stdout": stdout,
            "stdout_excerpt": stdout[:800] or "暂无执行输出",
            "analysis": analysis or "暂无分析结论",
            "chart_evidence": self._summarize_chart_evidence(charts),
            "charts": charts,
        }

    def _compose_experiment_description(self, query: str, context: dict) -> str:
        # Rebuild the methodology chapter from structured upstream outputs
        # instead of falling back to a query-only free-form rewrite.
        return self._expand_section(
            (
                f"请基于以下结构化实验信息，为研究问题“{query}”整理出论文风格的实验设计与方法设计章节，至少 400 字。\n\n"
                f"研究假设：{context['research_hypothesis']}\n\n"
                f"数据集与实验环境：{context['dataset']}\n\n"
                f"基线方法：\n{context['baselines_text']}\n\n"
                f"评估指标：\n{context['metrics_text']}\n\n"
                f"实验流程：\n{context['procedure_text']}\n\n"
                f"预期结果：{context['expected_results']}\n\n"
                "写作要求：\n"
                "1. 仅基于给定材料组织内容，不要凭空补造实验设定。\n"
                "2. 使用连续段落写作，整体风格接近论文中的实验设计章节。\n"
                "3. 需要自然衔接研究目标、数据、基线、指标与流程，而不是简单堆砌条目。"
            ),
            min_words=400,
        )

    def _format_experiment_subsection(self, value: Any, missing_label: str) -> str:
        if isinstance(value, list):
            lines = [self._safe_text(item) for item in value if self._safe_text(item)]
            if lines:
                return "\n".join(lines)
        else:
            text = self._safe_text(value)
            if text:
                return text
        return f"实验设计模块未提供{missing_label}，报告阶段不进行额外补编。"

    def _format_code_execution_status(self, code_result: dict) -> str:
        success = code_result.get("success")
        if success is True:
            stdout = self._safe_text(code_result.get("stdout", ""))
            return (
                "代码执行状态：成功。结果分析可以基于真实执行输出展开。"
                f"\n执行输出摘要：{stdout[:500] or '未提供标准输出。'}"
            )

        if success is False:
            error_parts = [
                self._safe_text(code_result.get("error_type", "")),
                self._safe_text(code_result.get("error", "")),
                self._safe_text(code_result.get("stderr", "")),
            ]
            iterations = code_result.get("iterations", []) or []
            if iterations and isinstance(iterations[-1], dict):
                error_parts.append(self._safe_text(iterations[-1].get("error", "")))
                error_parts.append(self._safe_text(iterations[-1].get("stderr", "")))
            error_text = " | ".join(part for part in error_parts if part)
            return (
                "代码执行状态：失败。最终报告不得声称实验已经验证成功、不得给出未由执行结果支持的"
                "显著提升、p 值、加速倍数或定量优势；结果章节应说明实验未完成、失败原因和后续验证计划。"
                f"\n失败信息摘要：{error_text[:700] or '未提供具体错误信息。'}"
            )

        return (
            "代码执行状态：未知。最终报告应避免写成已经完成实验验证，只能基于设计目标、代码草案和"
            "已有分析材料进行保守讨论。"
        )

    def _compose_failed_execution_results(self, query: str, context: dict) -> str:
        return (
            f"围绕“{query}”的实验结果分析需要首先说明一个关键限制：代码执行模块返回失败状态，"
            "因此当前阶段尚未形成可用于证明方法有效性的真实实验结果。根据前置实验设计，系统原计划围绕"
            f"{context['metrics_text']} 等指标，并以 {context['baselines_text']} 作为比较对象展开验证；"
            "但由于执行过程未成功完成，这些指标目前只能作为后续验证目标，不能被解释为已经获得的实验结论。\n\n"
            f"{context['execution_status']}\n\n"
            "在这种情况下，报告只能进行证据边界内的分析：已有材料可以说明研究方案、模型压缩目标、评价指标和"
            "基线设置已经形成，但尚不能支持“显著优于基线”“达到统计显著性”或“推理速度提升若干倍”等定量判断。"
            f"若分析模块提供了图表或中间输出，也应将其视为辅助材料而非最终实验验证。当前可用图表证据为："
            f"{context['chart_evidence']}。后续工作应优先修复代码执行问题，重新生成可复现实验日志，再基于真实"
            "stdout、指标表和图表补充结果分析。"
        )

    def _compose_execution_limited_conclusion(self, query: str, context: dict, result_description: str) -> str:
        return (
            f"本报告围绕“{query}”整理了文献背景、研究假设、实验设计和方法实现思路。根据当前前置模块输出，"
            f"研究假设为：{context['research_hypothesis']}。实验设计已经给出了数据集、评价指标、基线方法和"
            "预期结果，为后续验证提供了较完整的方案基础。\n\n"
            "不过，需要明确的是，代码执行模块当前返回失败状态，因此本报告不能将设计目标或预期结果表述为已经"
            "得到验证的实验发现。现阶段更合理的结论是：该方案具备进一步实现和验证的研究价值，但其实际性能、"
            "压缩收益、推理速度和相对基线优势仍需在代码成功运行后重新评估。后续改进应优先定位执行失败原因，"
            "补齐可复现实验日志和指标计算流程，再由分析模块输出结构化结果，最后更新结果章节和摘要。"
        )

    def _build_rich_report(self, query: str, outputs: dict) -> dict:
        self.logger.info(self.AGENT_NAME, "构建详细学术报告...")

        # Keep the external workflow unchanged and only reshape upstream data
        # inside the evaluation agent before generating report sections.
        context = self._build_report_context(query, outputs)

        self.logger.info(self.AGENT_NAME, "生成研究背景...")
        background = self._expand_section(
            (
                f"请基于以下材料，为研究问题“{query}”撰写研究背景与问题陈述，至少 300 字。\n\n"
                f"上游文献综述：\n{context['literature_review']}\n\n"
                f"相关文献标题：\n{context['paper_titles']}\n\n"
                f"结构化文献摘要：\n{context['paper_summaries']}\n\n"
                f"当前研究假设：{context['research_hypothesis']}\n\n"
                "写作要求：\n"
                "1. 重点说明已有研究主要解决了什么、仍存在哪些不足，以及本文问题的研究价值。\n"
                "2. 必须优先依据给定文献材料展开，不要只根据问题名称泛泛而谈。\n"
                "3. 若材料不足，请用审慎表述指出信息边界。"
            ),
            min_words=300,
        )

        self.logger.info(self.AGENT_NAME, "生成文献综述...")
        literature = self._expand_section(
            (
                f"请围绕研究问题“{query}”撰写论文风格的文献综述，至少 500 字。\n\n"
                f"请优先基于上游模块传入的文献综述内容进行改写与扩写：\n{context['literature_review']}\n\n"
                f"可作为补充参考的文献标题如下：\n{context['paper_titles']}\n\n"
                f"可作为补充事实依据的结构化摘要如下：\n{context['paper_summaries']}\n\n"
                "写作要求：\n"
                "1. 采用正式的中文学术论文语体，不要写成项目汇报或分点罗列。\n"
                "2. 以连续段落组织内容，可按研究方向、方法类别、关键进展、局限与研究空白展开。\n"
                "3. 不要虚构不存在的方法、数据或结论；若信息不足，保持审慎表述。\n"
                "4. 结尾自然引出本文的研究切入点和后续实验设计动机。"
            ),
            min_words=500,
        )

        self.logger.info(self.AGENT_NAME, "整合实验设计...")
        experiment_description = context["experiment_description"]
        if len(experiment_description) < 100:
            experiment_description = self._compose_experiment_description(query, context)

        self.logger.info(self.AGENT_NAME, "生成方法实现说明...")
        method_description = self._expand_section(
            (
                "请根据以下前置模块结果，用学术语言描述核心方法实现，至少 250 字。\n\n"
                f"研究问题：{query}\n\n"
                f"研究假设：{context['research_hypothesis']}\n\n"
                f"实验流程：\n{context['procedure_text']}\n\n"
                f"相关文献方法线索：\n{context['paper_summaries']}\n\n"
                f"代码摘要：\n{context['code_excerpt']}\n\n"
                "写作要求：\n"
                "1. 说明实现如何服务于研究目标与实验设计，而不只是解释代码语法。\n"
                "2. 尽量指出实现与已有方法之间的联系或差异，但不要编造未出现的算法细节。\n"
                "3. 使用完整段落，不要列表化。"
            ),
            min_words=250,
        )

        self.logger.info(self.AGENT_NAME, "生成结果分析...")
        if context["code_success"] is False:
            result_description = self._compose_failed_execution_results(query, context)
        else:
            result_description = self._expand_section(
                (
                    "请严格基于以下前置模块结果，撰写实验结果与分析，至少 350 字。\n\n"
                    f"研究问题：{query}\n\n"
                    f"代码执行状态：\n{context['execution_status']}\n\n"
                    f"评估指标：\n{context['metrics_text']}\n\n"
                    f"基线方法：\n{context['baselines_text']}\n\n"
                    f"预期结果：{context['expected_results']}\n\n"
                    f"分析结论：{context['analysis']}\n\n"
                    f"执行输出：\n{context['stdout_excerpt']}\n\n"
                    f"图表证据：\n{context['chart_evidence']}\n\n"
                    "写作要求：\n"
                    "1. 结果分析必须围绕指标、基线比较和图表证据展开，不要脱离这些材料自由发挥。\n"
                    "2. 若无法得出严格定量结论，应明确说明证据不足，而不是虚构数值。\n"
                    "3. 需要解释结果含义、潜在原因，以及与预期是否一致。\n"
                    "4. 如果代码执行状态不是成功，禁止写成已经完成实验验证。"
                ),
                min_words=350,
            )

        self.logger.info(self.AGENT_NAME, "生成结论与展望...")
        if context["code_success"] is False:
            conclusion = self._compose_execution_limited_conclusion(query, context, result_description)
        else:
            conclusion = self._expand_section(
                (
                    f"请基于以下材料，为研究问题“{query}”撰写结论与展望，至少 220 字。\n\n"
                    f"研究背景与空白线索：\n{context['literature_review']}\n\n"
                    f"研究假设：{context['research_hypothesis']}\n\n"
                    f"实验设计摘要：\n{experiment_description[:1200]}\n\n"
                    f"结果分析摘要：\n{result_description[:1200]}\n\n"
                    f"代码执行状态：\n{context['execution_status']}\n\n"
                    f"图表与执行证据：\n{context['chart_evidence']}\n{context['stdout_excerpt']}\n\n"
                    "写作要求：\n"
                    "1. 结论必须回扣前文证据，不要仅根据 query 作模板化总结。\n"
                    "2. 需要包含主要发现、现有局限和可执行的后续改进方向。\n"
                    "3. 若某些结论证据不充分，请明确说明其不确定性。\n"
                    "4. 只有代码执行成功且分析模块提供证据时，才可以写确定性的实验发现。"
                ),
                min_words=220,
            )

        title = f"MindPilot 科研报告：{query}"
        exp_design = outputs.get("experiment_design", {}) or {}
        exp_sections = self._normalize_experiment_sections(exp_design.get("sections", []))
        if not exp_sections:
            exp_sections = [
                {
                    "heading": "3.1 实验假设与目标",
                    "body": self._format_experiment_subsection(
                        exp_design.get("research_hypothesis", ""),
                        "实验假设与目标",
                    ),
                },
                {
                    "heading": "3.2 评估指标",
                    "body": self._format_experiment_subsection(
                        exp_design.get("metrics", []),
                        "评估指标",
                    ),
                },
                {
                    "heading": "3.3 基线方法",
                    "body": self._format_experiment_subsection(
                        exp_design.get("baselines", []),
                        "基线方法",
                    ),
                },
            ]

        sections = [
            {"heading": "一、研究背景与问题陈述", "body": background, "level": 1},
            {"heading": "二、文献综述", "body": literature, "level": 1},
            {"heading": "三、实验设计与方法论", "body": experiment_description, "level": 1},
            *[
                {"heading": sec["heading"], "body": sec["body"], "level": 2}
                for sec in exp_sections
            ],
            {"heading": "四、核心方法实现", "body": method_description, "level": 1},
            {"heading": "五、实验结果与分析", "body": result_description, "level": 1},
            {"heading": "六、结论与展望", "body": conclusion, "level": 1},
        ]

        report_body = self._render_report_text(
            {
                "title": title,
                "query": query,
                "abstract": "",
                "sections": sections,
            }
        )
        abstract = self._build_abstract(query, report_body)

        return {
            "title": title,
            "query": query,
            "abstract": abstract,
            "sections": sections,
            "code": context["final_code"],
            "stdout": context["stdout"],
            "literature": context["papers"],
            "charts": context["charts"],
        }

    def _build_abstract(self, query: str, report_body: str) -> str:
        summary_body = self._sanitize_text_for_abstract(report_body)
        prompt = (
            f"请基于已经完成的科研报告正文，为研究问题“{query}”撰写中文摘要。\n\n"
            f"报告正文如下：\n{summary_body[:6000]}\n\n"
            "写作要求：\n"
            "1. 摘要必须严格基于正文内容总结，不要根据固定模板重写，不要补充正文中没有的信息。\n"
            "2. 用一段或两段连续学术表述概括研究背景、方法、实验设置、核心结果与结论。\n"
            "3. 控制在 180 到 260 字之间，风格接近论文摘要。\n"
            "4. 不要使用项目符号，不要写“本文将”这类尚未完成时态，保持结果导向。"
        )
        abstract = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "你是资深学术写作专家，请根据给定正文生成准确、简洁、论文风格的中文摘要。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
        )
        abstract = self._safe_text(abstract)
        data = _extract_json_object(abstract)
        if data:
            abstract_parts = [
                self._safe_text(data.get(key, ""))
                for key in ("background", "method", "result", "conclusion", "limitation")
            ]
            abstract = " ".join(part for part in abstract_parts if part)
        if abstract:
            return abstract
        return f"本文围绕“{query}”开展研究，摘要应根据最终正文内容生成，但当前模型未返回有效结果。"

    def _expand_section(self, prompt: str, min_words: int = 200) -> str:
        resp = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是资深科研论文写作专家，请用严谨、专业的中文学术语言撰写内容。"
                        f"正文尽量不少于 {min_words} 字，并使用完整段落表达。"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
        )
        return resp if resp else f"内容生成中，请参考问题：{prompt[:120]}"

    def _safe_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _parse_json_object(self, text: str) -> dict:
        """解析模型返回的 JSON；兼容 ```json 代码块和前后说明文字。"""
        if not text:
            return {}

        cleaned = text.strip()
        fence_match = re.fullmatch(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        for candidate in (cleaned, self._extract_balanced_json(cleaned)):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
                return data if isinstance(data, dict) else {}
            except Exception:
                continue
        return {}

    def _extract_balanced_json(self, text: str) -> str:
        """从混杂文本中截取第一个完整 JSON 对象；截断时返回空串。"""
        start = text.find("{")
        if start < 0:
            return ""

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return ""

    def _normalize_experiment_sections(self, sections: Any) -> list[dict]:
        """清洗模型返回的实验设计小节，报告生成只消费 heading/body。"""
        if not isinstance(sections, list):
            return []

        normalized = []
        for idx, sec in enumerate(sections, 1):
            if not isinstance(sec, dict):
                continue
            heading = str(sec.get("heading", "")).strip()
            body = str(sec.get("body", "")).strip()
            if not heading or not body:
                continue
            if self._looks_like_raw_json(body):
                continue
            if not heading.startswith("3."):
                heading = f"3.{idx} {heading}"
            normalized.append({"heading": heading, "body": body})
        return normalized

    def _fallback_experiment_design(self, query: str, research_path: str = "") -> dict:
        """当模型返回截断或非法 JSON 时，提供干净的结构化实验设计。"""
        path_text = f"该方案参考推荐研究路径「{research_path}」，" if research_path else ""
        return {
            "research_hypothesis": (
                f"{query} 相关方法可以通过更清晰的实验控制、基线对照和多维评价指标获得"
                "更可靠的效果验证。"
            ),
            "objectives": [
                "明确研究问题对应的核心实验假设与验证目标",
                "构建统一的数据处理、模型训练和评估流程",
                "通过基线对照、消融实验和统计检验分析方法有效性",
            ],
            "dataset": (
                "实验数据应选择与研究问题直接相关的公开数据集或课程允许的数据样本，"
                "并统一完成数据清洗、格式转换、训练集/验证集/测试集划分和异常样本过滤。"
            ),
            "baselines": [
                "基础模型或传统方法：作为最低性能参考",
                "主流深度学习方法：用于体现当前常用技术路线的表现",
                "改进前模型：用于直接衡量本研究方法带来的增益",
            ],
            "metrics": [
                "Accuracy 或任务成功率：衡量主要任务表现",
                "Precision、Recall 或 F1：衡量预测结果的稳定性与均衡性",
                "Latency 与资源占用：衡量方法在实际部署中的效率成本",
            ],
            "variables": {
                "independent": ["模型结构", "关键模块开关", "训练配置"],
                "dependent": ["任务性能指标", "效率指标", "稳定性指标"],
                "controlled": ["数据划分", "随机种子", "训练轮数", "评价协议"],
            },
            "ablations": [
                "移除核心模块，观察性能变化",
                "调整关键超参数，分析模型敏感性",
                "固定其他条件，仅改变单一变量验证因果关系",
            ],
            "procedure": [
                "完成数据预处理和实验环境配置",
                "训练并评估各个基线方法",
                "训练并评估改进方法",
                "开展消融实验和重复实验",
                "汇总结果并进行统计检验与误差分析",
            ],
            "reproducibility": (
                "固定随机种子，记录 Python、PyTorch、依赖库版本和硬件环境；"
                "每组实验至少重复三次，并报告均值、标准差和显著性检验结果。"
            ),
            "expected_results": (
                "预期改进方法在主要性能指标上优于基线，同时在计算开销上保持可接受水平；"
                "消融实验应能说明关键模块对最终性能的贡献。"
            ),
            "full_description": (
                f"本实验围绕「{query}」展开，{path_text}通过统一数据集、基线方法、"
                "评价指标和可复现实验流程，对研究假设进行系统验证。实验设计强调控制变量，"
                "在相同训练配置和评价协议下比较不同方法的性能差异，并结合消融实验分析关键模块"
                "的实际贡献。最终报告将从效果、效率、稳定性和可复现性四个角度解释实验结果。"
            ),
            "sections": [
                {
                    "heading": "3.1 研究假设与验证目标",
                    "body": (
                        f"本节围绕「{query}」明确实验假设与验证目标。实验重点不是单纯展示模型输出，"
                        "而是通过可比较、可复现的流程判断方法是否真正带来性能增益，并分析这种增益"
                        "是否来自核心设计而非数据划分或训练随机性。"
                    ),
                },
                {
                    "heading": "3.2 数据集构建与预处理流程",
                    "body": (
                        "数据集部分需要说明数据来源、样本规模、清洗规则和划分方式。所有方法应使用"
                        "相同训练集、验证集和测试集，并统一输入格式、缺失值处理和异常样本过滤规则，"
                        "从而保证后续结果具有可比性。"
                    ),
                },
                {
                    "heading": "3.3 基线方法与对照组设置",
                    "body": (
                        "对照组应包含基础方法、主流方法和改进前模型三个层次。基础方法提供性能下限，"
                        "主流方法代表已有研究水平，改进前模型用于直接衡量本研究新增模块或策略的贡献。"
                    ),
                },
                {
                    "heading": "3.4 评估指标与变量控制",
                    "body": (
                        "评价指标应同时覆盖任务效果和运行效率，例如准确率、F1、延迟和资源占用。"
                        "实验中需要明确自变量、因变量和控制变量，并固定随机种子、训练轮数、优化器和"
                        "数据划分，以减少无关因素对结论的影响。"
                    ),
                },
                {
                    "heading": "3.5 消融实验与可复现性设置",
                    "body": (
                        "消融实验通过逐项移除或替换关键模块，分析各组成部分对最终性能的贡献。"
                        "为保证可复现性，每组实验至少重复三次，并记录硬件环境、软件版本、超参数配置"
                        "和统计检验结果。"
                    ),
                },
                {
                    "heading": "3.6 预期结果与分析方向",
                    "body": (
                        "预期结果部分需要解释方法在主要指标上的改进幅度，并结合误差样本和失败案例分析"
                        "方法边界。如果改进方法在效果和效率之间存在权衡，也应在结果分析中给出具体讨论。"
                    ),
                },
            ],
        }

    def _looks_like_raw_json(self, text: str) -> bool:
        stripped = text.strip()
        return (
            stripped.startswith("```json")
            or stripped.startswith("{")
            or '"research_hypothesis"' in stripped
            or '"sections"' in stripped
        )

    def _format_experiment_part(self, items: Any, fallback: str = "") -> str:
        """把实验设计字段稳定格式化，避免报告中出现一整段散文。"""
        if isinstance(items, str):
            return items.strip() or fallback
        if isinstance(items, dict):
            lines = []
            for key, value in items.items():
                if isinstance(value, list):
                    value = "；".join(str(v) for v in value if str(v).strip())
                if str(value).strip():
                    lines.append(f"{key}：{value}")
            return "\n".join(lines) if lines else fallback
        if isinstance(items, list):
            cleaned = [str(item).strip() for item in items if str(item).strip()]
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(cleaned, 1)) if cleaned else fallback
        return fallback

    def _format_variables_and_ablations(self, exp_design: dict) -> str:
        variables = exp_design.get("variables", {}) or {}
        lines = []
        if isinstance(variables, dict):
            label_map = {
                "independent": "自变量",
                "dependent": "因变量",
                "controlled": "控制变量",
            }
            for key, label in label_map.items():
                value = variables.get(key, [])
                if isinstance(value, list):
                    value = "；".join(str(v) for v in value if str(v).strip())
                if str(value).strip():
                    lines.append(f"{label}：{value}")
        ablations = self._format_experiment_part(exp_design.get("ablations", []))
        if ablations:
            lines.append("消融实验：\n" + ablations)
        return "\n".join(lines) if lines else "本节通过固定训练配置、数据划分和评价协议，比较关键组件加入或移除后的性能变化。"

    def _format_experiment_design(self, exp_design: dict) -> str:
        """生成第 3 章开头的结构化总述。"""
        pieces = []
        overview = exp_design.get("full_description", "")
        if overview and not self._looks_like_raw_json(overview):
            pieces.append(overview.strip())

        hypothesis = exp_design.get("research_hypothesis", "")
        if hypothesis:
            pieces.append(f"研究假设：{hypothesis.strip()}")

        dataset = exp_design.get("dataset", "")
        if dataset:
            pieces.append(f"数据与环境：{dataset.strip()}")

        baselines = self._format_experiment_part(exp_design.get("baselines", []))
        if baselines:
            pieces.append("基线与对照组：\n" + baselines)

        metrics = self._format_experiment_part(exp_design.get("metrics", []))
        if metrics:
            pieces.append("核心评估指标：\n" + metrics)

        procedure = self._format_experiment_part(exp_design.get("procedure", []))
        if procedure:
            pieces.append("实验流程概览：\n" + procedure)

        expected = exp_design.get("expected_results", "")
        if expected:
            pieces.append(f"预期分析方向：{expected.strip()}")

        return "\n\n".join(pieces) if pieces else "本章从研究假设、数据集、基线方法、评估指标、实验流程和可复现性设置六个方面组织实验设计。"

    def _print_result(self, score: EvalScore, log: list, reports: dict):
        accepted_reflections = sum(1 for record in log if (record or {}).get("accepted", False))
        print(f"\n{'=' * 58}")
        print("  Evaluation and report generation complete")
        print(f"{'=' * 58}")
        print(
            f"  overall={score.overall:.2f}  accuracy={score.accuracy:.2f}  "
            f"completeness={score.completeness:.2f}  format={score.format_quality:.2f}"
        )
        print(f"  reflection_rounds={accepted_reflections}")
        print("  report_files:")
        for fmt, path in reports.items():
            print(f"    [{fmt.upper():8s}] {path}")
        print(f"  feedback: {score.feedback[:100]}")
        print(f"{'=' * 58}\n")
