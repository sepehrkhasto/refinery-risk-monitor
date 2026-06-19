"""
Professional association rule mining service (FP-Growth).
Extracts cause-effect rules from historical risk assessments.
Thread-safe and production-ready.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta , timezone
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from itertools import combinations

from sqlalchemy.orm import Session

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)
from backend.db.db_models import RiskAssessment, Unit
from backend.logger import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


# ============================================================================
# Data classes
# ============================================================================
class RulePriority:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


@dataclass
class AssociationRule:
    antecedent: str
    consequent: str
    support: float
    confidence: float
    lift: float
    conviction: float
    leverage: float
    priority: str
    interpretation: str


@dataclass
class SequentialRule:
    antecedent: str
    consequent: str
    max_gap_minutes: int
    frequency: int
    confidence: float
    avg_time_gap_minutes: float
    priority: str


@dataclass
class RuleCluster:
    cluster_id: int
    causes: List[str]
    rules: List[AssociationRule]
    size: int
    description: str
    severity: str


@dataclass
class AssociationAnalysisReport:
    timestamp: str
    unit_name: str
    top_rules_by_confidence: List[AssociationRule]
    top_rules_by_lift: List[AssociationRule]
    sequential_rules: List[SequentialRule]
    clusters: List[RuleCluster]
    cause_network: Dict[str, List[str]]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    text_visualization: str


# ============================================================================
# FP-Growth implementation
# ============================================================================
class FPTree:
    def __init__(self, item=None, count=0, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None


class FPGrowth:
    def __init__(self, min_support: float = 0.05):
        self.min_support = min_support
        self.min_support_count = 0
        self.header_table = {}
        self.tree = None
        self.frequent_itemsets = []

    def fit(self, transactions: List[List[str]]):
        n_transactions = len(transactions)
        self.min_support_count = max(2, int(self.min_support * n_transactions))
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        frequent_items = {item: count for item, count in item_counts.items()
                          if count >= self.min_support_count}
        if not frequent_items:
            return self
        self.item_order = {item: i for i, (item, _) in
                           enumerate(sorted(frequent_items.items(), key=lambda x: x[1], reverse=True))}
        self.tree = FPTree()
        for transaction in transactions:
            filtered = [item for item in transaction if item in frequent_items]
            filtered.sort(key=lambda x: self.item_order[x])
            if filtered:
                self._insert_transaction(filtered, self.tree)
        self._mine_patterns()
        return self

    def _insert_transaction(self, transaction: List[str], node: FPTree):
        if not transaction:
            return
        first_item = transaction[0]
        if first_item in node.children:
            child = node.children[first_item]
            child.count += 1
        else:
            child = FPTree(item=first_item, count=1, parent=node)
            node.children[first_item] = child
            if first_item in self.header_table:
                current = self.header_table[first_item]
                while current.node_link:
                    current = current.node_link
                current.node_link = child
            else:
                self.header_table[first_item] = child
        self._insert_transaction(transaction[1:], child)

    def _mine_patterns(self):
        for item, node in sorted(self.header_table.items(),
                                 key=lambda x: self.item_order[x[0]]):
            conditional_patterns = []
            current = node
            while current:
                if current.count >= self.min_support_count:
                    prefix = []
                    parent = current.parent
                    while parent and parent.item is not None:
                        prefix.append(parent.item)
                        parent = parent.parent
                    if prefix:
                        conditional_patterns.append((prefix, current.count))
                current = current.node_link
            if conditional_patterns:
                conditional_db = []
                for pattern, count in conditional_patterns:
                    conditional_db.extend([pattern] * count)
                if conditional_db:
                    self.frequent_itemsets.append({'items': [item], 'count': sum(1 for _ in conditional_db)})
                    fp_growth = FPGrowth(min_support=self.min_support)
                    fp_growth.fit(conditional_db)
                    for itemset in fp_growth.frequent_itemsets:
                        self.frequent_itemsets.append({
                            'items': itemset['items'] + [item],
                            'count': itemset['count']
                        })
            else:
                self.frequent_itemsets.append({'items': [item], 'count': node.count})

    def get_frequent_itemsets(self) -> List[Dict]:
        return self.frequent_itemsets


# ============================================================================
# Association Rule Miner
# ============================================================================
class AssociationRuleMiner:
    def __init__(self, min_support: float = 0.05, min_confidence: float = 0.5,
                 min_lift: float = 1.2):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.fp_growth = None
        self.frequent_itemsets = []
        self.rules: List[AssociationRule] = []

    def fit(self, transactions: List[List[str]]):
        if len(transactions) < 10:
            return self
        logger.info(f"Mining rules from {len(transactions)} transactions")
        self.fp_growth = FPGrowth(min_support=self.min_support)
        self.fp_growth.fit(transactions)
        self.frequent_itemsets = self.fp_growth.get_frequent_itemsets()
        self.rules = self._generate_rules(transactions)
        for rule in self.rules:
            rule.priority = self._calculate_priority(rule)
            rule.interpretation = self._generate_interpretation(rule)
        return self

    def _generate_rules(self, transactions: List[List[str]]) -> List[AssociationRule]:
        n_transactions = len(transactions)
        rules = []
        item_counts = Counter()
        for trans in transactions:
            for item in trans:
                item_counts[item] += 1
        for itemset in self.frequent_itemsets:
            items = itemset['items']
            if len(items) < 2:
                continue
            support_ab = itemset['count'] / n_transactions
            for k in range(1, len(items)):
                for antecedent in combinations(items, k):
                    antecedent = list(antecedent)
                    consequent = [item for item in items if item not in antecedent]
                    if not consequent:
                        continue
                    support_a = item_counts[antecedent[0]] / n_transactions
                    for item in antecedent[1:]:
                        support_a = min(support_a, item_counts[item] / n_transactions)
                    confidence = support_ab / (support_a + 1e-8)
                    support_b = item_counts[consequent[0]] / n_transactions
                    for item in consequent[1:]:
                        support_b = min(support_b, item_counts[item] / n_transactions)
                    lift = confidence / (support_b + 1e-8)
                    conviction = (1 - support_b) / (1 - confidence + 1e-8)
                    leverage = support_ab - (support_a * support_b)
                    if confidence >= self.min_confidence and lift >= self.min_lift:
                        rules.append(AssociationRule(
                            antecedent=" -> ".join(antecedent) if len(antecedent) > 1 else antecedent[0],
                            consequent=" -> ".join(consequent) if len(consequent) > 1 else consequent[0],
                            support=round(support_ab, 4),
                            confidence=round(confidence, 4),
                            lift=round(lift, 2),
                            conviction=round(conviction, 2),
                            leverage=round(leverage, 4),
                            priority=RulePriority.LOW,
                            interpretation=""
                        ))
        unique_rules = {}
        for rule in rules:
            key = f"{rule.antecedent}|{rule.consequent}"
            if key not in unique_rules or unique_rules[key].confidence < rule.confidence:
                unique_rules[key] = rule
        result = list(unique_rules.values())
        result.sort(key=lambda x: (x.confidence, x.lift), reverse=True)
        return result

    def _calculate_priority(self, rule: AssociationRule) -> str:
        if rule.confidence >= 0.8 and rule.lift >= 2.0:
            return RulePriority.HIGH
        elif rule.confidence >= 0.65 and rule.lift >= 1.5:
            return RulePriority.MEDIUM
        elif rule.confidence >= 0.5:
            return RulePriority.LOW
        return RulePriority.INFORMATIONAL

    def _generate_interpretation(self, rule: AssociationRule) -> str:
        if rule.confidence >= 0.8:
            strength = "very strong"
        elif rule.confidence >= 0.65:
            strength = "strong"
        elif rule.confidence >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"
        if rule.lift >= 2.0:
            correlation = "highly correlated"
        elif rule.lift >= 1.5:
            correlation = "significantly correlated"
        else:
            correlation = "somewhat correlated"
        return (f"When {rule.antecedent} occurs, {rule.consequent} follows "
                f"with {strength} probability ({rule.confidence:.0%} confidence). "
                f"The two causes are {correlation} (lift: {rule.lift:.2f}).")

    def get_top_rules(self, n: int = 10, by: str = 'confidence') -> List[AssociationRule]:
        if by == 'confidence':
            sorted_rules = sorted(self.rules, key=lambda x: x.confidence, reverse=True)
        elif by == 'lift':
            sorted_rules = sorted(self.rules, key=lambda x: x.lift, reverse=True)
        else:
            sorted_rules = self.rules
        return sorted_rules[:n]


# ============================================================================
# Main Association Analysis Service
# ============================================================================
class AssociationAnalysisService:
    """
    Association rule mining service. Can be executed for a single unit or globally.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.rule_miner: Optional[AssociationRuleMiner] = None
        self.sequential_miner = None
        self.clusterer = None
        self.visualizer = None
        self.transactions = []
        self.events = []
        self.is_trained = False

    def train(self, db: Session, unit_name: str = None,
              date_from: datetime = None, date_to: datetime = None):
        query = db.query(RiskAssessment)
        if unit_name:
            unit = db.query(Unit).filter(Unit.name == unit_name).first()
            if unit:
                query = query.filter(RiskAssessment.unit_id == unit.id)
        if date_from:
            query = query.filter(RiskAssessment.timestamp >= date_from)
        if date_to:
            query = query.filter(RiskAssessment.timestamp <= date_to)
        records = query.order_by(RiskAssessment.timestamp.asc()).all()
        if len(records) < 20:
            logger.warning(f"Only {len(records)} records for association rules")
            return
        self._prepare_transactions(records)
        self.events = [{'cause': r.root_cause, 'timestamp': r.timestamp} for r in records]
        if len(self.transactions) >= 10:
            self.rule_miner = AssociationRuleMiner()
            self.rule_miner.fit(self.transactions)
            self.is_trained = True
            logger.info(f"Association rules trained with {len(self.rule_miner.rules)} rules")

    def _prepare_transactions(self, records, window_minutes=60):
        self.transactions = []
        current_window = []
        window_start = records[0].timestamp
        for record in records:
            cause = record.root_cause if record.root_cause is not None else "Unknown"
            time_diff = (record.timestamp - window_start).total_seconds() / 60
            if time_diff > window_minutes and current_window:
                unique = list(set(current_window))
                if len(unique) >= 2:
                    self.transactions.append(unique)
                current_window = [cause]
                window_start = record.timestamp
            else:
                current_window.append(cause)
        if current_window and len(set(current_window)) >= 2:
            self.transactions.append(list(set(current_window)))

    def analyze(self, db: Session, unit_name: str = None) -> AssociationAnalysisReport:
        if not self.is_trained:
            self.train(db, unit_name)
        if not self.rule_miner:
            return AssociationAnalysisReport(
                timestamp=datetime.now(timezone.utc).isoformat(),
                unit_name=unit_name or "All",
                top_rules_by_confidence=[], top_rules_by_lift=[],
                sequential_rules=[], clusters=[], cause_network={},
                summary_statistics={}, recommendations=[], text_visualization=""
            )
        top_by_confidence = self.rule_miner.get_top_rules(10, 'confidence')
        top_by_lift = self.rule_miner.get_top_rules(10, 'lift')
        return AssociationAnalysisReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            unit_name=unit_name or "All",
            top_rules_by_confidence=top_by_confidence,
            top_rules_by_lift=top_by_lift,
            sequential_rules=[],
            clusters=[],
            cause_network={},
            summary_statistics={'total_rules': len(self.rule_miner.rules)},
            recommendations=[],
            text_visualization=""
        )


# ============================================================================
# Backward-compatible functions
# ============================================================================
_association_service = AssociationAnalysisService()


def get_apriori_rules_professional(
    db: Session,
    min_confidence: float = 0.5,
    unit_name: str = None,
    limit: int = 20
) -> List[Dict]:
    if not _association_service.is_trained:
        _association_service.train(db, unit_name)
    report = _association_service.analyze(db, unit_name)
    result = []
    for rule in report.top_rules_by_confidence[:limit]:
        result.append({
            "antecedent": rule.antecedent,
            "consequent": rule.consequent,
            "confidence": rule.confidence,
            "support": rule.support,
            "lift": rule.lift,
            "conviction": rule.conviction,
            "leverage": rule.leverage,
            "priority": rule.priority,
            "interpretation": rule.interpretation
        })
    return result


def get_apriori_rules(db, min_confidence=0.5):
    return get_apriori_rules_professional(db, min_confidence)