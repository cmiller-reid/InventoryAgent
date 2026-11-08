"""
inventory_agent.py
==================
Supply Chain Analytics Inventory Decision Agent.

Implements the 7-stage decision architecture:
  Customer Question
       │
       ▼
 1. Identify Model       (EOQ, Quantity Discount, Safety Stock & ROP, Fixed-Period, Newsvendor)
       │
       ▼
 2. Extract Inputs       (Regex + Semantic Parser, Unit Normalizer, Missing Parameter Detector)
       │
       ▼
 3. Select Python Tool   (Maps to inventory_tools.py functions)
       │
       ▼
 4. Calculate            (Deterministic Python computation)
       │
       ▼
 5. Verify               (Mathematical consistency, operational bounds, edge-case checks)
       │
       ▼
 6. Explain              (Executive summary, math trace, trade-offs, managerial insights)
"""

import re
import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable

from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_newsvendor,
    calculate_safety_stock_rop,
    calculate_fixed_period_review,
    generate_eoq_cost_curve,
    generate_newsvendor_curve,
    EOQResult,
    QuantityDiscountResult,
    SafetyStockROPResult,
    PeriodicReviewResult,
    FixedPeriodResult,
    NewsvendorResult,
)


class InventoryModelType(str, Enum):
    EOQ = "Economic Order Quantity (EOQ)"
    QUANTITY_DISCOUNT = "Quantity Discount Model"
    SAFETY_STOCK_ROP = "Safety Stock & Reorder Point"
    FIXED_PERIOD_REVIEW = "Fixed-Period Review Model"
    NEWSVENDOR = "Newsvendor Model"
    UNKNOWN = "Unknown / General Query"


@dataclass
class VerificationCheck:
    name: str
    status: str  # "PASS", "WARNING", "FAIL"
    message: str


@dataclass
class AgentStepTrace:
    step_number: int
    step_name: str
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    customer_question: str
    model_type: InventoryModelType
    confidence: float
    identification_rationale: str
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str]
    selected_tool_name: str
    calculation_result: Optional[Dict[str, Any]]
    verification_checks: List[VerificationCheck]
    executive_recommendation: str
    detailed_explanation: str
    step_traces: List[AgentStepTrace]
    visualization_data: Optional[Dict[str, Any]] = None
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "customer_question": self.customer_question,
            "model_type": self.model_type.value,
            "confidence": self.confidence,
            "identification_rationale": self.identification_rationale,
            "extracted_inputs": self.extracted_inputs,
            "missing_inputs": self.missing_inputs,
            "selected_tool_name": self.selected_tool_name,
            "calculation_result": self.calculation_result,
            "verification_checks": [
                {"name": c.name, "status": c.status, "message": c.message}
                for c in self.verification_checks
            ],
            "executive_recommendation": self.executive_recommendation,
            "detailed_explanation": self.detailed_explanation,
            "step_traces": [
                {
                    "step_number": s.step_number,
                    "step_name": s.step_name,
                    "summary": s.summary,
                    "details": s.details,
                }
                for s in self.step_traces
            ],
            "success": self.success,
            "error_message": self.error_message,
        }


class InventoryDecisionAgent:
    """
    Intelligent Agent for Supply Chain Inventory Decision-Making.
    Follows strict 7-stage analytical execution.
    """

    def __init__(self):
        self.tool_mapping: Dict[InventoryModelType, Callable] = {
            InventoryModelType.EOQ: calculate_eoq,
            InventoryModelType.QUANTITY_DISCOUNT: calculate_quantity_discount,
            InventoryModelType.SAFETY_STOCK_ROP: calculate_safety_stock_rop,
            InventoryModelType.FIXED_PERIOD_REVIEW: calculate_fixed_period_review,
            InventoryModelType.NEWSVENDOR: calculate_newsvendor,
        }

    # =========================================================================
    # Stage 1: Identify Model
    # =========================================================================
    def identify_model(self, question: str) -> Tuple[InventoryModelType, float, str]:
        """
        Classifies user query into one of 5 inventory models using weighted semantic tokens.
        """
        text = question.lower()

        scores = {
            InventoryModelType.QUANTITY_DISCOUNT: 0.0,
            InventoryModelType.NEWSVENDOR: 0.0,
            InventoryModelType.FIXED_PERIOD_REVIEW: 0.0,
            InventoryModelType.SAFETY_STOCK_ROP: 0.0,
            InventoryModelType.EOQ: 0.0,
        }

        # Quantity Discount cues (high specificity)
        if any(w in text for w in ["quantity discount", "volume discount", "price break", "bracket", "discount tier", "price schedule"]):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 10.0
        if "discount" in text and ("order" in text or "price" in text or "tier" in text):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 6.0
        if re.search(r"\$\d+.*(?:for|if order).*?(?:unit|piece|order)", text):
            scores[InventoryModelType.QUANTITY_DISCOUNT] += 5.0

        # Newsvendor cues
        if any(w in text for w in ["newsvendor", "newsboy", "single period", "single-period", "critical fractile", "critical ratio"]):
            scores[InventoryModelType.NEWSVENDOR] += 12.0
        if any(w in text for w in ["salvage", "salvage value", "markdown", "liquidation", "perishable", "spoilage"]):
            scores[InventoryModelType.NEWSVENDOR] += 8.0
        if "overage" in text or "underage" in text:
            scores[InventoryModelType.NEWSVENDOR] += 8.0
        if ("selling price" in text or "sells for" in text) and ("cost" in text) and ("leftover" in text or "unsold" in text or "fashion" in text or "season" in text):
            scores[InventoryModelType.NEWSVENDOR] += 7.0

        # Fixed-Period Review cues
        if any(w in text for w in ["fixed period", "periodic review", "order up to", "order-up-to", "target inventory", "target level"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 12.0
        if any(w in text for w in ["review period", "review interval", "review every", "reviewed weekly", "reviewed monthly"]):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 9.0
        if "on hand" in text and ("on order" in text or "order up" in text or "review" in text):
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 6.0
        if "protection interval" in text:
            scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 10.0

        # Safety Stock & ROP cues
        if any(w in text for w in ["safety stock", "reorder point", "rop", "re-order point"]):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 11.0
        if ("lead time" in text or "lead-time" in text) and ("service level" in text or "csl" in text or "stockout" in text):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 7.0
        if "continuous review" in text or "(s, q)" in text or "(s,q)" in text:
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 10.0
        if ("std dev" in text or "standard deviation" in text or "sigma" in text) and ("lead time" in text) and ("review every" not in text):
            scores[InventoryModelType.SAFETY_STOCK_ROP] += 5.0

        # EOQ cues
        if any(w in text for w in ["eoq", "economic order quantity", "economic batch quantity"]):
            scores[InventoryModelType.EOQ] += 12.0
        if ("annual demand" in text or "demand" in text) and ("ordering cost" in text or "setup cost" in text) and ("holding cost" in text or "carrying cost" in text):
            scores[InventoryModelType.EOQ] += 8.0
        if "how much should we order" in text or "optimal order quantity" in text:
            scores[InventoryModelType.EOQ] += 4.0

        # Disambiguation rules
        # If both fixed period and safety stock score high, check for periodic review keywords
        if scores[InventoryModelType.FIXED_PERIOD_REVIEW] > 0 and scores[InventoryModelType.SAFETY_STOCK_ROP] > 0:
            if any(w in text for w in ["review period", "review every", "order up to", "order-up-to"]):
                scores[InventoryModelType.FIXED_PERIOD_REVIEW] += 5.0
            else:
                scores[InventoryModelType.SAFETY_STOCK_ROP] += 3.0

        # Pick winner
        best_model = max(scores, key=lambda k: scores[k])
        best_score = scores[best_model]

        if best_score < 3.0:
            return (
                InventoryModelType.UNKNOWN,
                0.20,
                "The question did not contain enough specific inventory keywords to decisively identify a model."
            )

        confidence = min(0.99, max(0.60, best_score / 14.0))
        rationale = f"Detected high-affinity terms for {best_model.value} (signal strength {best_score:.1f})."
        return best_model, round(confidence, 2), rationale

    # =========================================================================
    # Stage 2: Extract Inputs (Advanced Multi-Pass Semantic Extractor)
    # =========================================================================
    def _normalize_text(self, text: str) -> str:
        """Normalizes numbers, commas, and common spelling/hyphenation patterns."""
        # Normalize commas within numbers e.g. 10,000 -> 10000
        text = re.sub(r"(\d+),(\d+)", r"\1\2", text)
        # Normalize hyphens and abbreviations for inventory terms
        text = re.sub(r"\blead-time\b", "lead time", text, flags=re.IGNORECASE)
        text = re.sub(r"\bon-hand\b", "on hand", text, flags=re.IGNORECASE)
        text = re.sub(r"\bon-order\b", "on order", text, flags=re.IGNORECASE)
        text = re.sub(r"\bback-orders\b", "backorders", text, flags=re.IGNORECASE)
        text = re.sub(r"\bback-order\b", "backorder", text, flags=re.IGNORECASE)
        text = re.sub(r"\bstd[\.\s]+dev[\.]?\b", "standard deviation", text, flags=re.IGNORECASE)
        text = re.sub(r"\border-up-to\b", "order up to", text, flags=re.IGNORECASE)
        return text

    def _find_number_by_patterns(self, text: str, patterns: List[str]) -> Optional[float]:
        """Searches prioritized regex patterns to extract a float."""
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val_str = m.group(1).replace(",", "").strip()
                try:
                    return float(val_str)
                except ValueError:
                    continue
        return None

    def _find_number_by_proximity(
        self,
        text: str,
        target_terms: List[str],
        window: int = 5,
        exclude_terms: Optional[List[str]] = None
    ) -> Optional[float]:
        """
        Fall-back scanner: examines the entire question by scanning proximity around target keywords.
        Ensures parameters are extracted even with unusual word ordering or phrasing.
        """
        words = text.split()
        exclude_terms = [t.lower() for t in (exclude_terms or [])]
        target_terms = [t.lower() for t in target_terms]

        for i, word in enumerate(words):
            word_clean = re.sub(r"[^\w/]", "", word.lower())
            combos = [word_clean]
            if i + 1 < len(words):
                combos.append(word_clean + " " + re.sub(r"[^\w/]", "", words[i+1].lower()))
            if i + 2 < len(words):
                combos.append(word_clean + " " + re.sub(r"[^\w/]", "", words[i+1].lower()) + " " + re.sub(r"[^\w/]", "", words[i+2].lower()))

            matched_target = any(term in combo for term in target_terms for combo in combos)
            if matched_target:
                start = max(0, i - window)
                end = min(len(words), i + window + 3)
                for j in range(start, end):
                    token = words[j].replace(",", "").replace("$", "").rstrip("%").strip(".:;!?()")
                    try:
                        val = float(token)
                        # Ensure an exclude_term isn't closer to this number
                        is_excluded = False
                        if exclude_terms:
                            for ex in exclude_terms:
                                for k in range(max(0, j - 2), min(len(words), j + 3)):
                                    if ex in words[k].lower():
                                        is_excluded = True
                                        break
                                if is_excluded:
                                    break
                        if not is_excluded:
                            return val
                    except ValueError:
                        continue
        return None

    def _extract_eoq_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted = {}
        missing = []

        # 1. Ordering Cost (Recognize all user-specified terms & synonyms):
        # - ordering cost
        # - order cost
        # - setup cost
        # - setup expense
        # - cost per order
        # - cost of placing an order
        # - ordering/setup cost
        s = self._find_number_by_patterns(text, [
            r"(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an?\s+order|ordering[\/\s]+setup(?:\s+cost)?|cost\s+to\s+place\s+an?\s+order|order\s+placement\s+cost|ordering\s+and\s+setup\s+cost)\s*(?:of|is|are|:|=|at|equals?)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|\/\s*order|setup\s+expense|ordering\s+cost|order\s+cost|setup\s+cost)",
            r"(?:costs|incurs)\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+place\s+an?\s+order|each\s+time\s+an\s+order\s+is\s+placed)"
        ])
        if s is None:
            s = self._find_number_by_proximity(
                text,
                ["ordering cost", "order cost", "setup cost", "setup expense", "cost per order", "placing an order", "ordering/setup"],
                exclude_terms=["holding", "carrying", "unit price", "unit cost", "demand"]
            )

        # 2. Demand
        d = self._find_number_by_patterns(text, [
            r"(?:annual\s+demand|yearly\s+demand|demand\s+per\s+year|annual\s+usage|annual\s+sales|demand\s+is|demand\s+of)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|parts|cases)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|parts|cases)\s*(?:per|a|\/)\s*year",
            r"demand\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d is None:
            d = self._find_number_by_proximity(
                text,
                ["annual demand", "yearly demand", "demand", "annual usage"],
                exclude_terms=["cost", "order", "holding", "setup"]
            )

        # 3. Unit Cost / Purchase Price
        c = self._find_number_by_patterns(text, [
            r"(?:unit\s+cost|unit\s+purchase\s+price|purchase\s+price|unit\s+price|item\s+cost)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|\/\s*unit)(?!\s*(?:per\s+year|\/\s*year|annually))",
            r"costs\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:each|per\s+unit)"
        ])

        # 4. Holding Cost (fixed dollar amount or percentage)
        h = self._find_number_by_patterns(text, [
            r"(?:holding\s+cost|carrying\s+cost|holding\s+expense|carrying\s+expense|storage\s+cost)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)",
            r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|\/\s*unit\s*\/\s*year|\/\s*unit\s+annually|per\s+unit\s+annually)",
            r"holding\s*[:=]\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        h_rate = self._find_number_by_patterns(text, [
            r"(?:percentage\s+holding\s+cost|holding\s+cost\s+rate|carrying\s+cost\s+rate|holding\s+rate|carrying\s+rate|cost\s+of\s+capital|carrying\s+charge|holding\s+cost\s+is|carrying\s+cost\s+is)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding\s+cost|carrying\s+cost|holding\s+rate|carrying\s+rate|carrying\s+charge|of\s+unit\s+cost|of\s+unit\s+price)"
        ])

        if h is None and h_rate is not None and c is not None:
            rate_frac = h_rate / 100.0 if h_rate > 1.0 else h_rate
            h = rate_frac * c
        elif h is None:
            h = self._find_number_by_proximity(
                text,
                ["holding cost", "carrying cost", "holding expense", "carrying expense"],
                exclude_terms=["setup", "order", "demand"]
            )

        if d is not None: extracted["demand"] = d
        else: missing.append("demand (Annual demand D in units)")

        if s is not None: extracted["ordering_cost"] = s
        else: missing.append("ordering_cost (Ordering/Setup cost S in $)")

        if h is not None: extracted["holding_cost"] = h
        else: missing.append("holding_cost (Annual holding cost H per unit in $ or holding % + unit cost)")

        if c is not None: extracted["unit_cost"] = c

        return extracted, missing

    def _extract_quantity_discount_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted = {}
        missing = []

        # 1. Demand
        d = self._find_number_by_patterns(text, [
            r"(?:annual\s+demand|yearly\s+demand|demand\s+is|demand\s+of|demand)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|parts)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|parts)\s*(?:per|a|\/)\s*year",
            r"demand\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d is None:
            d = self._find_number_by_proximity(text, ["annual demand", "demand", "yearly demand"], exclude_terms=["cost", "order", "price"])

        # 2. Ordering cost (supports all 7 synonyms)
        s = self._find_number_by_patterns(text, [
            r"(?:ordering\s+cost|order\s+cost|setup\s+cost|setup\s+expense|cost\s+per\s+order|cost\s+of\s+placing\s+an?\s+order|ordering[\/\s]+setup\s+cost)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|\/\s*order|setup\s+expense|cost\s+per\s+order)"
        ])
        if s is None:
            s = self._find_number_by_proximity(text, ["ordering cost", "order cost", "setup cost", "setup expense", "cost per order", "placing an order"], exclude_terms=["holding", "unit price", "demand"])

        # 3. Holding cost: distinguish fixed holding cost vs percentage holding cost
        h_rate = self._find_number_by_patterns(text, [
            r"(?:percentage\s+holding\s+cost|holding\s+cost\s+rate|carrying\s+cost\s+rate|holding\s+rate|carrying\s+rate|holding\s+percentage|carrying\s+percentage|annual\s+carrying\s+charge|cost\s+of\s+capital)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding\s+cost|carrying\s+cost|holding\s+rate|carrying\s+rate|carrying\s+charge|of\s+unit\s+price|of\s+purchase\s+price)"
        ])
        h_const = self._find_number_by_patterns(text, [
            r"(?:fixed\s+holding\s+cost|constant\s+holding\s+cost|holding\s+cost\s+is|carrying\s+cost\s+is)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)",
            r"\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|\/\s*unit\s*\/\s*year|\/\s*unit\s+annually|per\s+unit\s+annually)"
        ])

        # 4. Unit purchase price (base price if specified)
        base_price = self._find_number_by_patterns(text, [
            r"(?:unit\s+purchase\s+price|base\s+price|regular\s+price|unit\s+price\s+without\s+discount)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])

        # 5. Quantity break discount tiers
        brackets = []
        tier_matches = re.findall(
            r"(\d+)\s*(?:-|to)\s*(\d+)\s*(?:units)?\s*[:@=at,]*\s*\$?([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE
        )
        for m in tier_matches:
            brackets.append({
                "min_qty": float(m[0]),
                "max_qty": float(m[1]),
                "unit_price": float(m[2])
            })

        open_matches = re.findall(
            r"(?:(?:over|>|>=|\+)\s*(\d+)|(\d+)\s*(?:\+|and\s+over|or\s+more))\s*(?:units)?\s*[:@=at,]*\s*\$?([0-9]+(?:\.[0-9]+)?)",
            text,
            re.IGNORECASE
        )
        for m in open_matches:
            min_q = float(m[0] if m[0] else m[1])
            price = float(m[2])
            brackets.append({
                "min_qty": min_q,
                "max_qty": None,
                "unit_price": price
            })

        if not brackets:
            q_breaks = re.findall(
                r"(?:quantity\s+break\s+at|discount\s+at|orders\s+of\s+at\s+least|orders\s+above)\s*(\d+)\s*(?:units)?\s*[\(\[]?\$?([0-9]+(?:\.[0-9]+)?)[\)\]]?",
                text,
                re.IGNORECASE
            )
            if q_breaks and base_price is not None:
                sorted_breaks = sorted([(float(q), float(p)) for q, p in q_breaks], key=lambda x: x[0])
                current_min = 1.0
                current_price = base_price
                for break_q, break_price in sorted_breaks:
                    brackets.append({"min_qty": current_min, "max_qty": break_q - 1, "unit_price": current_price})
                    current_min = break_q
                    current_price = break_price
                brackets.append({"min_qty": current_min, "max_qty": None, "unit_price": current_price})

        if d is not None: extracted["demand"] = d
        else: missing.append("demand (Annual demand D)")

        if s is not None: extracted["ordering_cost"] = s
        else: missing.append("ordering_cost (Ordering cost S)")

        if h_rate is not None:
            extracted["holding_cost_rate"] = h_rate / 100.0 if h_rate > 1.0 else h_rate
        elif h_const is not None:
            extracted["constant_holding_cost"] = h_const
        else:
            missing.append("holding_cost_rate or constant_holding_cost (e.g. 20% holding rate or $2/unit)")

        if brackets:
            unique_tiers = {b["min_qty"]: b for b in brackets}
            extracted["discount_brackets"] = sorted(unique_tiers.values(), key=lambda x: x["min_qty"])
        else:
            missing.append("discount_brackets (Price discount tiers)")

        return extracted, missing

    def _extract_rop_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted = {}
        missing = []

        # 1. Distinguish lead-time standard deviation first
        lt_std = self._find_number_by_patterns(text, [
            r"(?:lead\s*time|delivery\s*time)\s*(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of\s+)?(?:the\s+)?(?:lead\s*time|delivery\s*time)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"lead\s*time\s*varies\s*(?:with\s*)?(?:a\s*)?(?:standard\s+deviation|std\s*dev)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"sigma[_\s]*l\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if lt_std is None:
            lt_std = 0.0

        # 2. Distinguish demand standard deviation
        d_std = self._find_number_by_patterns(text, [
            r"(?:demand|daily\s+demand|weekly\s+demand)\s*(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of\s+)?(?:the\s+)?(?:demand|daily\s+demand|weekly\s+demand)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"demand\s*varies\s*(?:with\s*)?(?:a\s*)?(?:standard\s+deviation|std\s*dev)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"sigma[_\s]*d\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d_std is None:
            d_std = self._find_number_by_patterns(text, [
                r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*units"
            ])
        if d_std is None:
            d_std = 0.0

        # 3. Distinguish average lead time (ensuring not matched to lead-time std dev)
        lt = self._find_number_by_patterns(text, [
            r"(?:average|mean)\s*(?:replenishment\s*)?(?:lead\s*time|delivery\s*time)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"(?:lead\s*time|delivery\s*time|replenishment\s*time)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)",
            r"takes\s*(?:an\s*average\s*of\s*)?([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:to\s*arrive|for\s*delivery|lead\s*time)?",
            r"lead\s*time\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if lt is None:
            lt = self._find_number_by_proximity(
                text,
                ["lead time", "delivery time", "replenishment time", "takes"],
                exclude_terms=["standard deviation", "std dev", "sigma", "demand"]
            )

        # 4. Distinguish average demand (ensuring not matched to demand std dev)
        d_avg = self._find_number_by_patterns(text, [
            r"(?:average\s+demand|mean\s+demand|daily\s+demand|weekly\s+demand|demand\s+averages)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?",
            r"demand\s*(?:is|of|:|=)\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|parts)?\s*(?:per\s*(?:day|week|month|period)|\/\s*(?:day|week|month|period))",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)\s*(?:per\s*day|\/\s*day|daily|per\s*week|\/\s*week)",
            r"demand\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d_avg is None:
            d_avg = self._find_number_by_proximity(
                text,
                ["average demand", "mean demand", "daily demand", "weekly demand", "demand"],
                exclude_terms=["standard deviation", "std dev", "sigma", "lead time"]
            )

        # 5. Service level
        sl = self._find_number_by_patterns(text, [
            r"(?:cycle\s+service\s+level|service\s+level|csl|in[- ]stock\s+probability|target\s+service\s+level|desired\s+service\s+level)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+service\s+level|service\s+level|csl|in[- ]stock)"
        ])
        if sl is not None and sl > 1.0:
            sl = sl / 100.0
        elif sl is None:
            sl = 0.95

        h = self._find_number_by_patterns(text, [
            r"(?:holding\s+cost|carrying\s+cost)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])

        if lt is not None: extracted["lead_time"] = lt
        else: missing.append("lead_time (Average replenishment lead time L)")

        if d_avg is not None: extracted["avg_demand"] = d_avg
        else: missing.append("avg_demand (Average demand per period d)")

        extracted["std_demand"] = d_std
        extracted["std_lead_time"] = lt_std
        extracted["service_level"] = sl
        if h is not None: extracted["holding_cost"] = h

        return extracted, missing

    def _extract_periodic_review_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted = {}
        missing = []

        # 1. Review Period
        p = self._find_number_by_patterns(text, [
            r"(?:review\s+period|review\s+interval|review\s+every|order\s+every|periodic\s+review\s+interval|period\s+p|review\s+cycle)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"reviewed\s+every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)",
            r"review\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if p is None:
            p = self._find_number_by_proximity(text, ["review period", "review interval", "review every", "reviewed every", "order every"])

        # 2. Lead Time
        lt = self._find_number_by_patterns(text, [
            r"(?:replenishment\s+)?(?:lead\s+time|delivery\s+time)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"takes\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:to\s*arrive|for\s*delivery)?"
        ])
        if lt is None:
            lt = self._find_number_by_proximity(text, ["lead time", "delivery time"], exclude_terms=["review", "demand"])

        # 3. Average Demand
        d_avg = self._find_number_by_patterns(text, [
            r"(?:average\s+demand|mean\s+demand|daily\s+demand|weekly\s+demand|demand\s+is|demand\s+of)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)\s*(?:per\s*day|\/\s*day|per\s*week|\/\s*week)"
        ])
        if d_avg is None:
            d_avg = self._find_number_by_proximity(text, ["average demand", "mean demand", "daily demand", "demand"], exclude_terms=["review", "lead time", "standard deviation", "std dev"])

        # 4. Demand Standard Deviation
        d_std = self._find_number_by_patterns(text, [
            r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s*dev\s+of\s+demand|demand\s+std\s*dev|standard\s+deviation|std\s*dev|sigma)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d_std is None:
            d_std = 0.0

        # 5. Service Level
        sl = self._find_number_by_patterns(text, [
            r"(?:service\s+level|cycle\s+service\s+level|csl|in[- ]stock\s+probability|target\s+service\s+level|desired\s+service\s+level)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:service\s+level|csl|in[- ]stock)"
        ])
        if sl is not None and sl > 1.0:
            sl = sl / 100.0
        elif sl is None:
            sl = 0.95

        # 6. Distinguish On-Hand, On-Order, Backorders, Inventory Position
        on_hand = self._find_number_by_patterns(text, [
            r"(?:on\s+hand\s+inventory|inventory\s+on\s+hand|on\s+hand|current\s+inventory|current\s+stock|stock\s+on\s+hand)\s*(?:of|is|are|total|equal|equals|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*(?:on\s+hand|in\s+stock)"
        ])
        on_order = self._find_number_by_patterns(text, [
            r"(?:on\s+order\s+inventory|inventory\s+on\s+order|on\s+order|pipeline\s+inventory|scheduled\s+receipts)\s*(?:of|is|are|total|equal|equals|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*on\s+order"
        ])
        backorders = self._find_number_by_patterns(text, [
            r"(?:backorders|back-orders|backordered\s+units|backordered\s+items|backlog)\s*(?:of|is|are|total|equal|equals|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*(?:are\s+)?backordered",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units|items)?\s*backorders"
        ])
        direct_pos = self._find_number_by_patterns(text, [
            r"(?:inventory\s+position|net\s+inventory|total\s+inventory\s+position)\s*(?:of|is|are|total|equal|equals|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)"
        ])

        if on_hand is None and direct_pos is not None:
            on_hand = direct_pos
        elif on_hand is None:
            on_hand = 0.0

        if on_order is None:
            on_order = 0.0
        if backorders is None:
            backorders = 0.0

        if p is not None: extracted["review_period"] = p
        else: missing.append("review_period (Review interval P)")

        if lt is not None: extracted["lead_time"] = lt
        else: missing.append("lead_time (Replenishment lead time L)")

        if d_avg is not None: extracted["avg_demand"] = d_avg
        else: missing.append("avg_demand (Average demand per period)")

        extracted["std_demand"] = d_std
        extracted["service_level"] = sl
        extracted["current_inventory"] = on_hand
        extracted["on_order"] = on_order
        extracted["backorders"] = backorders

        return extracted, missing

    def _extract_newsvendor_inputs(self, text: str) -> Tuple[Dict[str, Any], List[str]]:
        extracted = {}
        missing = []

        # 1. Distinguish selling price
        price = self._find_number_by_patterns(text, [
            r"(?:selling\s+price|retail\s+price|revenue\s+per\s+unit|price\s+per\s+unit)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"sells?\s+(?:[a-zA-Z0-9_\-\s]{1,30}?\s*)?(?:for|at)\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"price\s*(?:of|is|:|=|at)\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        if price is None:
            price = self._find_number_by_proximity(text, ["selling price", "retail price", "sells for"], exclude_terms=["cost", "salvage", "buy", "produce"])

        # 2. Distinguish purchase cost
        cost = self._find_number_by_patterns(text, [
            r"(?:purchase\s+cost|unit\s+cost|acquisition\s+cost|wholesale\s+cost|production\s+cost)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"(?:that\s+)?costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to\s+buy|to\s+produce|each|per\s+unit)?",
            r"costs?\s*(?:of|is|:|=|at)\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        if cost is None:
            cost = self._find_number_by_proximity(text, ["purchase cost", "unit cost", "wholesale cost", "cost to buy", "cost to produce"], exclude_terms=["sell", "retail", "price", "salvage"])

        # 3. Distinguish salvage value
        salvage = self._find_number_by_patterns(text, [
            r"(?:salvage\s+value|markdown\s+price|markdown\s+salvage\s+value|markdown\s+salvage|liquidation\s+value|scrap\s+value)\s*(?:of|is|for|at|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"salvages?\s*(?:for|at|of|is|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"unsold\s+(?:[a-zA-Z0-9_\-\s]{1,20}?\s*)?(?:salvage|sell\s+for|liquidated\s+for)\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        if salvage is None:
            salvage = self._find_number_by_proximity(text, ["salvage value", "markdown", "salvage for", "scrap value"], exclude_terms=["sell for", "retail", "purchase cost"])
        if salvage is None:
            salvage = 0.0

        # 4. Distinguish underage / shortage cost and overage / excess cost
        underage = self._find_number_by_patterns(text, [
            r"(?:underage\s+cost|shortage\s+cost|cost\s+of\s+underage|cost\s+of\s+shortage)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        overage = self._find_number_by_patterns(text, [
            r"(?:overage\s+cost|excess\s+cost|cost\s+of\s+overage|cost\s+of\s+excess)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        penalty = self._find_number_by_patterns(text, [
            r"(?:penalty\s+cost|goodwill\s+loss|loss\s+of\s+goodwill)\s*(?:of|is|:|=|at)?\s*\$?([0-9]+(?:\.[0-9]+)?)"
        ])
        if penalty is None:
            penalty = 0.0

        # If underage and overage are given directly instead of price and cost:
        if underage is not None and overage is not None:
            cost = overage
            salvage = 0.0
            price = cost + underage

        # 5. Distinguish mean demand
        d_mean = self._find_number_by_patterns(text, [
            r"(?:mean\s+demand|expected\s+demand|average\s+demand|demand\s+has\s+a\s+mean|mean\s+of)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"normally\s+distributed\s+with\s*(?:a\s*)?mean\s*(?:of)?\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d_mean is None:
            d_mean = self._find_number_by_proximity(text, ["mean demand", "expected demand", "average demand", "mean of", "mean"], exclude_terms=["standard deviation", "std dev", "sigma", "price", "cost"])

        # 6. Distinguish demand standard deviation
        d_std = self._find_number_by_patterns(text, [
            r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|std\s*dev\s+of\s+demand|demand\s+std\s*dev)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:standard\s+deviation|std\s*dev|sigma)\s*(?:of|is|:|=|at)?\s*([0-9]+(?:\.[0-9]+)?)"
        ])
        if d_std is None:
            d_std = self._find_number_by_proximity(text, ["standard deviation", "std dev", "sigma"], exclude_terms=["mean", "price", "cost"])

        if price is not None: extracted["selling_price"] = price
        else: missing.append("selling_price (Selling price p or underage cost Cu)")

        if cost is not None: extracted["purchase_cost"] = cost
        else: missing.append("purchase_cost (Purchase cost c or overage cost Co)")

        extracted["salvage_value"] = salvage
        extracted["penalty_cost"] = penalty

        if d_mean is not None: extracted["demand_mean"] = d_mean
        else: missing.append("demand_mean (Mean demand mu)")

        if d_std is not None: extracted["demand_std"] = d_std
        else: missing.append("demand_std (Demand standard deviation sigma)")

        return extracted, missing

    def extract_inputs(self, question: str, model_type: InventoryModelType) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts model-specific numerical values and parameters from the text.
        Examines the entire question with terminology synonyms and multi-pass entity extraction.
        Returns (extracted_inputs, missing_inputs).
        """
        normalized_text = self._normalize_text(question)

        if model_type == InventoryModelType.EOQ:
            return self._extract_eoq_inputs(normalized_text)
        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            return self._extract_quantity_discount_inputs(normalized_text)
        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            return self._extract_rop_inputs(normalized_text)
        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            return self._extract_periodic_review_inputs(normalized_text)
        elif model_type == InventoryModelType.NEWSVENDOR:
            return self._extract_newsvendor_inputs(normalized_text)
        else:
            return {}, ["Unsupported model"]

    # =========================================================================
    # Stage 3: Select Python Tool
    # =========================================================================
    def select_tool(self, model_type: InventoryModelType) -> Tuple[Optional[Callable], str]:
        """Maps identified model to the pure Python analytical tool."""
        tool = self.tool_mapping.get(model_type)
        tool_name = tool.__name__ if tool else "None"
        return tool, tool_name

    # =========================================================================
    # Stage 4: Calculate
    # =========================================================================
    def calculate(self, tool: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the mathematical tool function."""
        result = tool(**inputs)
        return result.to_dict()

    # =========================================================================
    # Stage 5: Verify
    # =========================================================================
    def verify(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        calc_result: Dict[str, Any]
    ) -> List[VerificationCheck]:
        """
        Extracts deterministic verification checks computed directly inside the Python tool.
        """
        raw_checks = calc_result.get("verification_checks", [])
        checks: List[VerificationCheck] = []
        for c in raw_checks:
            checks.append(VerificationCheck(
                name=c.get("check_name", "Verification Check"),
                status=c.get("status", "PASS"),
                message=c.get("details", "")
            ))
        return checks

    # =========================================================================
    # Stage 6: Explain (8-Point Student Verification Framework)
    # =========================================================================
    def _render_student_verification_sections(
        self,
        calc_result: Dict[str, Any],
        verification_checks: List[VerificationCheck]
    ) -> str:
        """
        Renders the complete 8-part student verification breakdown directly from
        the deterministic Python tool output.
        """
        # 1. Inputs Extracted
        inputs_data = calc_result.get("inputs_extracted", {})
        inputs_lines = []
        for k, v in inputs_data.items():
            if v is not None:
                if k == "discount_brackets":
                    brackets_str = ", ".join([
                        f"[{b['min_qty']:,.0f} - {b.get('max_qty', 'inf') if b.get('max_qty') is not None else 'inf'} @ ${b['unit_price']:.2f}]"
                        for b in v
                    ])
                    inputs_lines.append(f"- **{k}**: {brackets_str}")
                else:
                    inputs_lines.append(f"- **{k}**: {v}")
        inputs_md = "\n".join(inputs_lines)

        # 2. Variable Definitions
        var_defs = calc_result.get("variable_definitions", {})
        var_defs_md = "\n".join([f"- **{k}**: {v}" for k, v in var_defs.items()])

        # 3. Formula
        formulas = calc_result.get("formula", {})
        if isinstance(formulas, dict):
            formulas_md = "\n".join([f"- **{k.replace('_', ' ').title()}**: `{v}`" for k, v in formulas.items()])
        else:
            formulas_md = f"```\n{formulas}\n```"

        # 4. Formula with Substituted Values
        sub_formulas = calc_result.get("formula_with_substituted_values", {})
        if isinstance(sub_formulas, dict):
            sub_md = "\n".join([f"- **{k.replace('_', ' ').title()}**:\n  `{v}`" for k, v in sub_formulas.items()])
        else:
            sub_md = f"```\n{sub_formulas}\n```"

        # 5. Intermediate Calculations
        inter_calcs = calc_result.get("intermediate_calculations", {})
        inter_rows = []
        for k, v in inter_calcs.items():
            if k not in ["tier_evaluations"]:
                inter_rows.append(f"| `{k}` | **{v}** |")
        inter_table = (
            "| Intermediate Metric | Computed Value |\n| :--- | :--- |\n" + "\n".join(inter_rows)
            if inter_rows else "See detailed tier evaluations above."
        )

        # 6. Final Answer
        final_ans = calc_result.get("final_answer", {})
        final_md = "\n".join([f"- **{k.replace('_', ' ').title()}**: **{v}**" for k, v in final_ans.items()])

        # 7. Verification Checks
        checks_md = "\n".join([
            f"- **[{c.status}] {c.name}**: {c.message}" for c in verification_checks
        ])

        # 8. Assumptions
        assumptions = calc_result.get("assumptions", [])
        assumptions_md = "\n".join([f"{i+1}. {a}" for i, a in enumerate(assumptions)])

        return f"""
---
### 🎓 Student Verification Breakdown (8 Mandatory Sections)

#### 1. Inputs Extracted
{inputs_md}

#### 2. Variable Definitions
{var_defs_md}

#### 3. Formula
{formulas_md}

#### 4. Formula with Substituted Values
{sub_md}

#### 5. Intermediate Calculations
{inter_table}

#### 6. Final Answer
{final_md}

#### 7. Verification Checks
{checks_md}

#### 8. Assumptions
{assumptions_md}
"""

    def explain(
        self,
        model_type: InventoryModelType,
        inputs: Dict[str, Any],
        calc_result: Dict[str, Any],
        verification_checks: List[VerificationCheck]
    ) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        """
        Generates executive recommendation, detailed mathematical explanations,
        and visual curve data.
        """
        vis_data = None

        if model_type == InventoryModelType.EOQ:
            eoq = calc_result["eoq"]
            eoq_rnd = calc_result["eoq_rounded"]
            tot_cost = calc_result["total_annual_inventory_cost"]
            orders_yr = calc_result["orders_per_year"]
            cycle_days = calc_result["cycle_time_days"]

            vis_data = {
                "type": "eoq_curve",
                "data": generate_eoq_cost_curve(
                    demand=inputs["demand"],
                    ordering_cost=inputs["ordering_cost"],
                    holding_cost=inputs["holding_cost"],
                    eoq=eoq
                ),
                "optimal_q": eoq
            }

            rec = (
                f"**Recommendation**: Place orders for **{eoq_rnd:,} units** approximately every "
                f"**{cycle_days:.0f} work days** ({orders_yr:.1f} orders/year). "
                f"This minimizes total annual inventory holding and setup costs to **${tot_cost:,.2f}**."
            )

            exp = f"""### Analytical Step-by-Step Breakdown

1. **Economic Order Quantity Formula**:
   $$\\text{{EOQ}} = \\sqrt{{\\frac{{2 \\cdot D \\cdot S}}{{H}}}}$$
   - Annual Demand ($D$) = **{inputs['demand']:,}** units
   - Ordering / Setup Cost ($S$) = **${inputs['ordering_cost']:,.2f}** per order
   - Annual Holding Cost ($H$) = **${inputs['holding_cost']:,.2f}** per unit/year
   
2. **Calculation**:
   $$\\text{{EOQ}} = \\sqrt{{\\frac{{2 \\cdot {inputs['demand']:,} \\cdot {inputs['ordering_cost']}}}{{{inputs['holding_cost']}}}}} = \\sqrt{{{2 * inputs['demand'] * inputs['ordering_cost'] / inputs['holding_cost']:,.2f}}} = \\mathbf{{{eoq:,.2f}}} \\text{{ units}}$$
   - Practical Rounded Order Batch: **{eoq_rnd:,} units**

3. **Annual Cost Breakdown**:
   - **Annual Ordering Cost**: $\\frac{{D}}{{Q}} \\cdot S = \\frac{{{inputs['demand']:,}}}{{{eoq:,.2f}}} \\cdot {inputs['ordering_cost']} = \\mathbf{{\\${calc_result['annual_ordering_cost']:,.2f}}}$
   - **Annual Holding Cost**: $\\frac{{Q}}{{2}} \\cdot H = \\frac{{{eoq:,.2f}}}{{2}} \\cdot {inputs['holding_cost']} = \\mathbf{{\\${calc_result['annual_holding_cost']:,.2f}}}$
   - **Total Annual Inventory Cost**: **${tot_cost:,.2f}**
   {f"- **Annual Purchase Cost**: ${calc_result['annual_purchase_cost']:,.2f}\\n   - **Total Cost (Inventory + Goods)**: **${calc_result['total_annual_cost']:,.2f}**" if calc_result.get('total_annual_cost') else ""}

4. **Operational Cadence**:
   - **Order Frequency**: {orders_yr:.2f} orders per year.
   - **Cycle Time**: {cycle_days:.1f} operating days between replenishments.
"""

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            opt_tier = calc_result["optimal_tier_index"]
            opt_q = calc_result["optimal_order_quantity"]
            opt_price = calc_result["optimal_unit_price"]
            opt_cost = calc_result["optimal_total_annual_cost"]
            savings = calc_result["total_savings"]
            savings_pct = calc_result["savings_percentage"]

            rec = (
                f"**Recommendation**: Order in **Tier {opt_tier}** batches of **{opt_q:,.0f} units** "
                f"at the discounted price of **${opt_price:,.2f}/unit**. "
                f"Total annual expenditure is **${opt_cost:,.2f}**, delivering **${savings:,.2f} ({savings_pct:.1f}%)** "
                f"in net savings compared to the baseline tier."
            )

            rows = []
            for t in calc_result["tier_evaluations"]:
                max_str = f"{t['max_quantity']:,.0f}" if t.get("max_quantity") is not None else "inf"
                feas_str = "Yes" if t["is_raw_eoq_feasible"] else "No (Use Cutoff)"
                rows.append(
                    f"| Tier {t['tier_index']} | {t['min_quantity']:,.0f} - {max_str} | "
                    f"${t['unit_price']:.2f} | {t['raw_eoq']:,.1f} | {feas_str} | "
                    f"{t['candidate_quantity']:,.0f} | ${t['total_annual_cost']:,.2f} |"
                )
            tier_rows = "\n".join(rows)

            exp = f"""### Analytical Step-by-Step Breakdown

1. **Discount Tier Evaluation Matrix**:
| Tier | Quantity Range | Unit Price | Unconstrained EOQ | Feasible? | Candidate Order Q | Total Annual Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{tier_rows}

2. **Decision Logic**:
   - For each price tier, unconstrained $EOQ_i = \\sqrt{{\\frac{{2DS}}{{H_i}}}}$ is computed.
   - If $EOQ_i$ falls below the discount threshold, ordering exactly the minimum threshold quantity yields the lowest cost in that bracket.
   - Evaluating total annual costs (Purchase + Ordering + Holding) confirms **Tier {opt_tier}** with **{opt_q:,.0f} units** minimizes global total cost.
"""

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            ss = calc_result["safety_stock"]
            ss_rnd = calc_result["safety_stock_rounded"]
            rop = calc_result["reorder_point"]
            rop_rnd = calc_result["reorder_point_rounded"]
            csl = calc_result["service_level"]
            z = calc_result["z_score"]
            ltd = calc_result["expected_lead_time_demand"]
            sigma_ltd = calc_result["std_lead_time_demand"]

            rec = (
                f"**Recommendation**: Set the **Reorder Point (ROP) at {rop_rnd:,} units** "
                f"(which includes a **Safety Stock buffer of {ss_rnd:,} units**). "
                f"Whenever inventory drops to {rop_rnd:,} units, place a replenishment order immediately to achieve a **{csl*100:.1f}% Cycle Service Level**."
            )

            exp = f"""### Analytical Step-by-Step Breakdown

1. **Model Scenario**: **{calc_result['scenario']}**
   - Average Demand ($d$) = **{inputs['avg_demand']:,}** units/period
   - Average Lead Time ($L$) = **{inputs['lead_time']}** periods
   - Demand Std Dev ($\\sigma_d$) = **{inputs.get('std_demand', 0):,}**
   - Lead Time Std Dev ($\\sigma_L$) = **{inputs.get('std_lead_time', 0):,}**

2. **Lead Time Demand & Uncertainty**:
   - **Expected Lead Time Demand**: $\\mu_{{DLT}} = d \\cdot L = {inputs['avg_demand']} \\cdot {inputs['lead_time']} = \\mathbf{{{ltd:,.1f}}} \\text{{ units}}$
   - **Lead Time Demand Std Dev**: $\\sigma_{{DLT}} = \\mathbf{{{sigma_ltd:,.2f}}} \\text{{ units}}$

3. **Safety Stock & Reorder Point**:
   - **Target Service Level**: {csl*100:.1f}% $\\implies Z = \\Phi^{{-1}}({csl}) = \\mathbf{{{z:.4f}}}$
   - **Safety Stock**: $SS = Z \\cdot \\sigma_{{DLT}} = {z:.4f} \\cdot {sigma_ltd:.2f} = \\mathbf{{{ss:,.2f}}} \\approx \\mathbf{{{ss_rnd:,}}} \\text{{ units}}$
   - **Reorder Point**: $ROP = \\mu_{{DLT}} + SS = {ltd:,.1f} + {ss:,.2f} = \\mathbf{{{rop:,.2f}}} \\approx \\mathbf{{{rop_rnd:,}}} \\text{{ units}}$
   - **Cycle Stockout Risk**: **{calc_result['stockout_risk_per_cycle']*100:.2f}%** per replenishment cycle.
"""

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            p = calc_result["review_period"]
            lt = calc_result["lead_time"]
            prot_int = calc_result["protection_interval"]
            target_inv = calc_result["target_inventory_level_rounded"]
            rec_q = calc_result["recommended_order_quantity_rounded"]
            ss = calc_result["safety_stock_rounded"]
            net_pos = calc_result["current_inventory_position"]

            rec = (
                f"**Recommendation**: Review inventory every **{p:.0f} periods**. "
                f"Target Order-Up-To Level is **{target_inv:,} units** (including **{ss:,} units safety stock**). "
                f"With current inventory position of **{net_pos:,.0f} units**, place an order for **{rec_q:,} units** today."
            )

            exp = f"""### Analytical Step-by-Step Breakdown

1. **Protection Interval**:
   - Review Interval ($P$) = **{p}** periods
   - Replenishment Lead Time ($L$) = **{lt}** periods
   - Total Protection Horizon = $P + L = \\mathbf{{{prot_int:.1f}}} \\text{{ periods}}$

2. **Target Inventory Level ($M$ or $S$)**:
   - Expected Demand over Protection Interval = $d \\cdot (P + L) = {inputs['avg_demand']} \\cdot {prot_int} = \\mathbf{{{calc_result['expected_demand_protection_interval']:,.1f}}}$
   - Std Dev over Protection Interval = $\\sigma_d \\sqrt{{P + L}} = {inputs['std_demand']} \\cdot \\sqrt{{{prot_int}}} = \\mathbf{{{calc_result['std_demand_protection_interval']:,.2f}}}$
   - Safety Stock ($SS$) = $Z \\cdot \\sigma_{{P+L}} = {calc_result['z_score']:.4f} \\cdot {calc_result['std_demand_protection_interval']:.2f} = \\mathbf{{{calc_result['safety_stock']:,.2f}}} \\approx \\mathbf{{{ss:,}}}$
   - **Order-Up-To Level ($M$)**: **{target_inv:,} units**

3. **Order Quantity Determination**:
   - Current Inventory Position = On-Hand ({inputs.get('current_inventory', 0)}) + On-Order ({inputs.get('on_order', 0)}) = **{net_pos:,.0f} units**
   - **Order Quantity ($Q$)**: $\\max(0, M - \\text{{Position}}) = {target_inv} - {net_pos:,.0f} = \\mathbf{{{rec_q:,}}} \\text{{ units}}$
"""

        elif model_type == InventoryModelType.NEWSVENDOR:
            q_opt = calc_result["optimal_order_quantity"]
            q_opt_rnd = calc_result["optimal_order_quantity_rounded"]
            cr = calc_result["critical_fractile"]
            cu = calc_result["underage_cost"]
            co = calc_result["overage_cost"]
            profit = calc_result["expected_profit"]
            lost = calc_result["expected_lost_sales"]
            leftover = calc_result["expected_leftover_inventory"]

            vis_data = {
                "type": "newsvendor_curve",
                "data": generate_newsvendor_curve(
                    selling_price=inputs["selling_price"],
                    purchase_cost=inputs["purchase_cost"],
                    salvage_value=inputs.get("salvage_value", 0.0),
                    demand_mean=inputs["demand_mean"],
                    demand_std=inputs["demand_std"],
                    penalty_cost=inputs.get("penalty_cost", 0.0)
                ),
                "optimal_q": q_opt
            }

            rec = (
                f"**Recommendation**: Order **{q_opt_rnd:,} units** for this period. "
                f"The Critical Fractile is **{cr:.2%}**, resulting in an expected maximum net profit of "
                f"**${profit:,.2f}** with an in-stock probability of **{calc_result['in_stock_probability']:.1%}**."
            )

            exp = f"""### Analytical Step-by-Step Breakdown

1. **Underage & Overage Economic Analysis**:
   - Selling Price ($p$) = **${inputs['selling_price']:,.2f}**
   - Purchase / Production Cost ($c$) = **${inputs['purchase_cost']:,.2f}**
   - Salvage Value ($s$) = **${inputs.get('salvage_value', 0):,.2f}**
   - **Cost of Underage ($C_u$)**: $p - c = \\mathbf{{\\${cu:,.2f}}}$ (Profit margin lost per unmet unit)
   - **Cost of Overage ($C_o$)**: $c - s = \\mathbf{{\\${co:,.2f}}}$ (Loss incurred per unsold unit)

2. **Critical Fractile & Optimal Order Quantity**:
   $$\\text{{Critical Fractile (CR)}} = \\frac{{C_u}}{{C_u + C_o}} = \\frac{{{cu:,.2f}}}{{{cu:,.2f} + {co:,.2f}}} = \\mathbf{{{cr:.4f}}} \\; ({cr*100:.2f}\\%)$$
   - Standard Normal Quantile: $Z = \\Phi^{{-1}}({cr:.4f}) = \\mathbf{{{calc_result['z_score']:.4f}}}$
   - Mean Demand ($\\mu$) = **{inputs['demand_mean']:,}**, Std Dev ($\\sigma$) = **{inputs['demand_std']:,}**
   $$Q^* = \\mu + Z \\cdot \\sigma = {inputs['demand_mean']:,} + ({calc_result['z_score']:.4f} \\cdot {inputs['demand_std']:,}) = \\mathbf{{{q_opt:,.2f}}} \\approx \\mathbf{{{q_opt_rnd:,}}} \\text{{ units}}$$

3. **Expected Probabilistic Performance**:
   - **Expected Sales**: {calc_result['expected_sales']:,.1f} units
   - **Expected Lost Sales**: {lost:,.1f} units
   - **Expected Leftover Inventory**: {leftover:,.1f} units
   - **Expected Gross Revenue**: ${calc_result['expected_revenue']:,.2f}
   - **Expected Salvage Revenue**: ${calc_result['expected_salvage_revenue']:,.2f}
   - **Expected Total Cost**: ${calc_result['expected_purchase_cost']:,.2f}
   - **Expected Maximum Profit**: **${profit:,.2f}**
"""
        else:
            rec = "Unable to provide specific recommendation for unknown model."
            exp = "Please provide more details regarding your inventory question."

        if calc_result and ("inputs_extracted" in calc_result or "inputs" in calc_result):
            exp = exp.strip() + "\n\n" + self._render_student_verification_sections(calc_result, verification_checks)

        return rec, exp, vis_data

    # =========================================================================
    # Full Agent Orchestration Pipeline
    # =========================================================================
    def process_query(self, question: str) -> AgentResponse:
        """
        Executes the full 7-step Supply Chain Inventory Decision Agent architecture:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
        """
        step_traces: List[AgentStepTrace] = []

        # Step 0: Record Customer Question
        step_traces.append(AgentStepTrace(
            step_number=0,
            step_name="Customer Question Received",
            summary="User submitted natural-language query.",
            details={"raw_question": question}
        ))

        # Step 1: Identify Model
        model_type, confidence, id_rationale = self.identify_model(question)
        step_traces.append(AgentStepTrace(
            step_number=1,
            step_name="Identify Model",
            summary=f"Identified model as '{model_type.value}' with {confidence:.0%} confidence.",
            details={
                "model_type": model_type.value,
                "confidence": confidence,
                "rationale": id_rationale
            }
        ))

        if model_type == InventoryModelType.UNKNOWN:
            return AgentResponse(
                customer_question=question,
                model_type=model_type,
                confidence=confidence,
                identification_rationale=id_rationale,
                extracted_inputs={},
                missing_inputs=["All parameters"],
                selected_tool_name="None",
                calculation_result=None,
                verification_checks=[],
                executive_recommendation="Could not identify a supported inventory model from your query.",
                detailed_explanation=(
                    "Supported models are:\n"
                    "1. Economic Order Quantity (EOQ)\n"
                    "2. Quantity Discount Model\n"
                    "3. Safety Stock and Reorder Point (Continuous Review)\n"
                    "4. Fixed-Period Review Model (Order-Up-To)\n"
                    "5. Newsvendor Model (Single-Period)\n\n"
                    "Please include key parameters like demand, costs, lead time, or service levels."
                ),
                step_traces=step_traces,
                success=False,
                error_message="Unsupported or ambiguous inventory question."
            )

        # Step 2: Extract Inputs
        extracted_inputs, missing_inputs = self.extract_inputs(question, model_type)
        step_traces.append(AgentStepTrace(
            step_number=2,
            step_name="Extract Inputs",
            summary=f"Extracted {len(extracted_inputs)} parameters; {len(missing_inputs)} missing.",
            details={
                "extracted_inputs": extracted_inputs,
                "missing_inputs": missing_inputs
            }
        ))

        if missing_inputs:
            missing_str = "\n- " + "\n- ".join(missing_inputs)
            return AgentResponse(
                customer_question=question,
                model_type=model_type,
                confidence=confidence,
                identification_rationale=id_rationale,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing_inputs,
                selected_tool_name="None",
                calculation_result=None,
                verification_checks=[],
                executive_recommendation=f"Missing required parameters to solve for **{model_type.value}**.",
                detailed_explanation=f"To complete the analysis, please provide:{missing_str}",
                step_traces=step_traces,
                success=False,
                error_message=f"Missing inputs: {', '.join(missing_inputs)}"
            )

        # Step 3: Select Python Tool
        tool, tool_name = self.select_tool(model_type)
        step_traces.append(AgentStepTrace(
            step_number=3,
            step_name="Select Python Tool",
            summary=f"Mapped to pure Python analytical engine: {tool_name}() in inventory_tools.py.",
            details={"tool_name": tool_name}
        ))

        # Step 4: Calculate
        try:
            calc_result = self.calculate(tool, extracted_inputs)
            step_traces.append(AgentStepTrace(
                step_number=4,
                step_name="Calculate",
                summary="Successfully executed deterministic numerical computations.",
                details={"calculation_output": calc_result}
            ))
        except Exception as e:
            return AgentResponse(
                customer_question=question,
                model_type=model_type,
                confidence=confidence,
                identification_rationale=id_rationale,
                extracted_inputs=extracted_inputs,
                missing_inputs=[],
                selected_tool_name=tool_name,
                calculation_result=None,
                verification_checks=[],
                executive_recommendation="Calculation failed due to numerical or parameter validation error.",
                detailed_explanation=f"Error during calculation: {str(e)}",
                step_traces=step_traces,
                success=False,
                error_message=str(e)
            )

        # Step 5: Verify
        verification_checks = self.verify(model_type, extracted_inputs, calc_result)
        step_traces.append(AgentStepTrace(
            step_number=5,
            step_name="Verify",
            summary=f"Conducted {len(verification_checks)} mathematical and operational validity checks.",
            details={"checks": [{"name": c.name, "status": c.status, "message": c.message} for c in verification_checks]}
        ))

        # Step 6: Explain
        rec, exp, vis_data = self.explain(model_type, extracted_inputs, calc_result, verification_checks)
        step_traces.append(AgentStepTrace(
            step_number=6,
            step_name="Explain",
            summary="Synthesized executive recommendation, step-by-step math derivation, and strategic advice.",
            details={"recommendation": rec}
        ))

        return AgentResponse(
            customer_question=question,
            model_type=model_type,
            confidence=confidence,
            identification_rationale=id_rationale,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            selected_tool_name=tool_name,
            calculation_result=calc_result,
            verification_checks=verification_checks,
            executive_recommendation=rec,
            detailed_explanation=exp,
            step_traces=step_traces,
            visualization_data=vis_data,
            success=True
        )
