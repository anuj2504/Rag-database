"""
Domain-Specific Acronym and Terminology Database.

Key insight from enterprise RAG:
- Acronym confusion kills semantic search accuracy
- "CAR" means different things in different contexts
- Context-aware expansion is CRITICAL for specialized domains

This module provides:
- Domain-specific acronym databases
- Context-aware expansion
- Query enhancement with term expansion
- Custom terminology management
"""
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Domain(Enum):
    """Supported domains."""
    LEGAL = "legal"
    TAX = "tax"
    BUILDING = "building"
    FINANCIAL = "financial"
    GENERAL = "general"


@dataclass
class TermDefinition:
    """Definition of a term or acronym."""
    term: str                      # The acronym or term
    expansion: str                 # Full expansion
    domain: Domain                 # Primary domain
    definition: str = ""           # Optional definition
    aliases: List[str] = field(default_factory=list)  # Alternative forms
    related_terms: List[str] = field(default_factory=list)
    context_keywords: List[str] = field(default_factory=list)  # Keywords that suggest this meaning


class AcronymDatabase:
    """
    Domain-specific acronym and terminology database.

    Handles context-aware expansion for domain-specific terms.
    """

    def __init__(self):
        self.terms: Dict[str, List[TermDefinition]] = {}
        self.domain_terms: Dict[Domain, Set[str]] = {d: set() for d in Domain}

        # Load default databases
        self._load_default_databases()

    def _load_default_databases(self):
        """Load built-in domain databases."""
        self._load_legal_terms()
        self._load_tax_terms()
        self._load_building_terms()
        self._load_financial_terms()

    def _load_legal_terms(self):
        """Legal/contract terminology."""
        legal_terms = [
            # Contract terms
            TermDefinition(
                term="NDA",
                expansion="Non-Disclosure Agreement",
                domain=Domain.LEGAL,
                definition="A legal contract establishing confidential relationship",
                aliases=["non-disclosure", "confidentiality agreement"],
                context_keywords=["confidential", "proprietary", "secret"]
            ),
            TermDefinition(
                term="LOI",
                expansion="Letter of Intent",
                domain=Domain.LEGAL,
                definition="Document outlining agreement before formal contract",
                context_keywords=["intent", "preliminary", "proposal"]
            ),
            TermDefinition(
                term="SOW",
                expansion="Statement of Work",
                domain=Domain.LEGAL,
                definition="Document defining project scope and deliverables",
                context_keywords=["scope", "deliverables", "services"]
            ),
            TermDefinition(
                term="MSA",
                expansion="Master Services Agreement",
                domain=Domain.LEGAL,
                definition="Umbrella contract for ongoing services",
                context_keywords=["services", "master", "umbrella"]
            ),
            TermDefinition(
                term="SLA",
                expansion="Service Level Agreement",
                domain=Domain.LEGAL,
                definition="Agreement defining service performance standards",
                context_keywords=["service", "level", "uptime", "performance"]
            ),
            TermDefinition(
                term="IP",
                expansion="Intellectual Property",
                domain=Domain.LEGAL,
                definition="Creations of the mind protected by law",
                context_keywords=["patent", "copyright", "trademark", "trade secret"]
            ),
            TermDefinition(
                term="LLC",
                expansion="Limited Liability Company",
                domain=Domain.LEGAL,
                definition="Business structure combining corporate and partnership features",
                context_keywords=["company", "business", "entity"]
            ),
            TermDefinition(
                term="JV",
                expansion="Joint Venture",
                domain=Domain.LEGAL,
                definition="Business arrangement where parties agree to pool resources",
                context_keywords=["venture", "partnership", "collaboration"]
            ),
            TermDefinition(
                term="M&A",
                expansion="Mergers and Acquisitions",
                domain=Domain.LEGAL,
                definition="Consolidation of companies or assets",
                aliases=["merger", "acquisition"],
                context_keywords=["merger", "acquire", "takeover", "buyout"]
            ),
            TermDefinition(
                term="ADR",
                expansion="Alternative Dispute Resolution",
                domain=Domain.LEGAL,
                definition="Methods of resolving disputes outside of court",
                context_keywords=["arbitration", "mediation", "dispute"]
            ),
        ]

        for term_def in legal_terms:
            self.add_term(term_def)

    def _load_tax_terms(self):
        """Tax/IRC terminology."""
        tax_terms = [
            TermDefinition(
                term="IRC",
                expansion="Internal Revenue Code",
                domain=Domain.TAX,
                definition="The domestic portion of federal statutory tax law",
                context_keywords=["tax", "federal", "revenue", "section"]
            ),
            TermDefinition(
                term="IRS",
                expansion="Internal Revenue Service",
                domain=Domain.TAX,
                definition="U.S. government agency responsible for tax collection",
                context_keywords=["tax", "federal", "audit", "filing"]
            ),
            TermDefinition(
                term="AGI",
                expansion="Adjusted Gross Income",
                domain=Domain.TAX,
                definition="Gross income minus specific deductions",
                context_keywords=["income", "deduction", "taxable"]
            ),
            TermDefinition(
                term="FICA",
                expansion="Federal Insurance Contributions Act",
                domain=Domain.TAX,
                definition="Payroll tax for Social Security and Medicare",
                context_keywords=["payroll", "social security", "medicare", "withholding"]
            ),
            TermDefinition(
                term="FUTA",
                expansion="Federal Unemployment Tax Act",
                domain=Domain.TAX,
                definition="Federal tax for unemployment compensation",
                context_keywords=["unemployment", "employer", "payroll"]
            ),
            TermDefinition(
                term="AMT",
                expansion="Alternative Minimum Tax",
                domain=Domain.TAX,
                definition="Parallel tax system to ensure minimum tax payment",
                context_keywords=["minimum", "alternative", "preference"]
            ),
            TermDefinition(
                term="QBI",
                expansion="Qualified Business Income",
                domain=Domain.TAX,
                definition="Income eligible for Section 199A deduction",
                context_keywords=["business", "deduction", "pass-through", "199A"]
            ),
            TermDefinition(
                term="SALT",
                expansion="State and Local Taxes",
                domain=Domain.TAX,
                definition="State and local taxes deductible on federal returns",
                context_keywords=["state", "local", "deduction", "property"]
            ),
            TermDefinition(
                term="RMD",
                expansion="Required Minimum Distribution",
                domain=Domain.TAX,
                definition="Minimum amount that must be withdrawn from retirement accounts",
                context_keywords=["retirement", "distribution", "IRA", "401k"]
            ),
            TermDefinition(
                term="QSBS",
                expansion="Qualified Small Business Stock",
                domain=Domain.TAX,
                definition="Stock eligible for capital gains exclusion under Section 1202",
                context_keywords=["stock", "capital gains", "exclusion", "1202"]
            ),
        ]

        for term_def in tax_terms:
            self.add_term(term_def)

    def _load_building_terms(self):
        """Building code terminology."""
        building_terms = [
            TermDefinition(
                term="IBC",
                expansion="International Building Code",
                domain=Domain.BUILDING,
                definition="Model building code used throughout the United States",
                context_keywords=["building", "construction", "code", "occupancy"]
            ),
            TermDefinition(
                term="IRC",
                expansion="International Residential Code",
                domain=Domain.BUILDING,
                definition="Building code for one- and two-family dwellings",
                context_keywords=["residential", "dwelling", "house", "home"]
            ),
            TermDefinition(
                term="IFC",
                expansion="International Fire Code",
                domain=Domain.BUILDING,
                definition="Model code for fire prevention and safety",
                context_keywords=["fire", "safety", "sprinkler", "alarm"]
            ),
            TermDefinition(
                term="ADA",
                expansion="Americans with Disabilities Act",
                domain=Domain.BUILDING,
                definition="Civil rights law prohibiting discrimination based on disability",
                context_keywords=["accessibility", "disability", "wheelchair", "ramp"]
            ),
            TermDefinition(
                term="HVAC",
                expansion="Heating, Ventilation, and Air Conditioning",
                domain=Domain.BUILDING,
                definition="Technology for indoor environmental comfort",
                context_keywords=["heating", "cooling", "ventilation", "air"]
            ),
            TermDefinition(
                term="MEP",
                expansion="Mechanical, Electrical, and Plumbing",
                domain=Domain.BUILDING,
                definition="Building systems for utilities and services",
                context_keywords=["mechanical", "electrical", "plumbing", "systems"]
            ),
            TermDefinition(
                term="NFPA",
                expansion="National Fire Protection Association",
                domain=Domain.BUILDING,
                definition="Organization publishing fire safety codes and standards",
                context_keywords=["fire", "protection", "safety", "code"]
            ),
            TermDefinition(
                term="OSHA",
                expansion="Occupational Safety and Health Administration",
                domain=Domain.BUILDING,
                definition="Federal agency ensuring workplace safety",
                context_keywords=["safety", "workplace", "hazard", "violation"]
            ),
            TermDefinition(
                term="LEED",
                expansion="Leadership in Energy and Environmental Design",
                domain=Domain.BUILDING,
                definition="Green building certification program",
                context_keywords=["green", "sustainable", "energy", "environmental"]
            ),
            TermDefinition(
                term="R-value",
                expansion="Thermal Resistance Value",
                domain=Domain.BUILDING,
                definition="Measure of insulation's ability to resist heat flow",
                aliases=["R value", "Rvalue"],
                context_keywords=["insulation", "thermal", "energy", "efficiency"]
            ),
        ]

        for term_def in building_terms:
            self.add_term(term_def)

    def _load_financial_terms(self):
        """Financial terminology."""
        financial_terms = [
            TermDefinition(
                term="EBITDA",
                expansion="Earnings Before Interest, Taxes, Depreciation, and Amortization",
                domain=Domain.FINANCIAL,
                definition="Measure of company's operating performance",
                context_keywords=["earnings", "operating", "profit", "margin"]
            ),
            TermDefinition(
                term="ROI",
                expansion="Return on Investment",
                domain=Domain.FINANCIAL,
                definition="Ratio of net profit to investment cost",
                context_keywords=["return", "investment", "profit", "ratio"]
            ),
            TermDefinition(
                term="ROE",
                expansion="Return on Equity",
                domain=Domain.FINANCIAL,
                definition="Net income divided by shareholders' equity",
                context_keywords=["equity", "return", "shareholders", "ratio"]
            ),
            TermDefinition(
                term="P/E",
                expansion="Price to Earnings Ratio",
                domain=Domain.FINANCIAL,
                definition="Stock price divided by earnings per share",
                aliases=["PE", "PE ratio"],
                context_keywords=["price", "earnings", "stock", "valuation"]
            ),
            TermDefinition(
                term="EPS",
                expansion="Earnings Per Share",
                domain=Domain.FINANCIAL,
                definition="Net income divided by outstanding shares",
                context_keywords=["earnings", "share", "profit", "diluted"]
            ),
            TermDefinition(
                term="GAAP",
                expansion="Generally Accepted Accounting Principles",
                domain=Domain.FINANCIAL,
                definition="Standard framework for financial accounting",
                context_keywords=["accounting", "standards", "financial", "reporting"]
            ),
            TermDefinition(
                term="SEC",
                expansion="Securities and Exchange Commission",
                domain=Domain.FINANCIAL,
                definition="U.S. government agency regulating securities markets",
                context_keywords=["securities", "regulation", "filing", "disclosure"]
            ),
            TermDefinition(
                term="IPO",
                expansion="Initial Public Offering",
                domain=Domain.FINANCIAL,
                definition="First sale of stock by a company to the public",
                context_keywords=["public", "stock", "offering", "shares"]
            ),
            TermDefinition(
                term="DCF",
                expansion="Discounted Cash Flow",
                domain=Domain.FINANCIAL,
                definition="Valuation method based on future cash flow projections",
                context_keywords=["valuation", "cash flow", "discount", "present value"]
            ),
            TermDefinition(
                term="CAGR",
                expansion="Compound Annual Growth Rate",
                domain=Domain.FINANCIAL,
                definition="Rate of return for investment over time",
                context_keywords=["growth", "annual", "rate", "compound"]
            ),
        ]

        for term_def in financial_terms:
            self.add_term(term_def)

    def add_term(self, term_def: TermDefinition):
        """Add a term to the database."""
        key = term_def.term.upper()

        if key not in self.terms:
            self.terms[key] = []

        self.terms[key].append(term_def)
        self.domain_terms[term_def.domain].add(key)

        # Also index aliases
        for alias in term_def.aliases:
            alias_key = alias.upper()
            if alias_key not in self.terms:
                self.terms[alias_key] = []
            self.terms[alias_key].append(term_def)

    def get_term(
        self,
        term: str,
        context: str = None,
        domain_hint: Domain = None
    ) -> Optional[TermDefinition]:
        """
        Get term definition with context-aware disambiguation.

        Args:
            term: The term/acronym to look up
            context: Surrounding text for disambiguation
            domain_hint: Preferred domain if known

        Returns:
            Best matching TermDefinition or None
        """
        key = term.upper()

        if key not in self.terms:
            return None

        definitions = self.terms[key]

        if len(definitions) == 1:
            return definitions[0]

        # Multiple definitions - need to disambiguate
        if domain_hint:
            for defn in definitions:
                if defn.domain == domain_hint:
                    return defn

        if context:
            # Score each definition by context keyword matches
            best_score = 0
            best_defn = definitions[0]
            context_lower = context.lower()

            for defn in definitions:
                score = sum(
                    1 for kw in defn.context_keywords
                    if kw in context_lower
                )
                if score > best_score:
                    best_score = score
                    best_defn = defn

            return best_defn

        # Default to first definition
        return definitions[0]

    def expand_query(
        self,
        query: str,
        domain: Domain = None
    ) -> Tuple[str, List[str]]:
        """
        Expand acronyms in a query.

        Args:
            query: Original query
            domain: Domain context

        Returns:
            Tuple of (expanded_query, list of expansions applied)
        """
        expanded = query
        expansions = []

        # Find potential acronyms (2-6 uppercase letters)
        acronym_pattern = r'\b([A-Z]{2,6})\b'

        for match in re.finditer(acronym_pattern, query):
            acronym = match.group(1)
            defn = self.get_term(acronym, context=query, domain_hint=domain)

            if defn:
                # Add expansion in parentheses
                expansion = f"{acronym} ({defn.expansion})"
                expanded = expanded.replace(acronym, expansion, 1)
                expansions.append(f"{acronym} → {defn.expansion}")

        return expanded, expansions

    def get_related_terms(
        self,
        term: str,
        domain: Domain = None
    ) -> List[str]:
        """Get related terms for query expansion."""
        defn = self.get_term(term, domain_hint=domain)
        if defn:
            return defn.related_terms + defn.aliases
        return []

    def detect_domain(self, text: str) -> Domain:
        """Detect likely domain from text content."""
        text_lower = text.lower()

        domain_scores = {d: 0 for d in Domain}

        for domain, terms in self.domain_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    domain_scores[domain] += 1

        # Additional keyword detection
        domain_keywords = {
            Domain.LEGAL: ['agreement', 'contract', 'party', 'whereas', 'hereby'],
            Domain.TAX: ['tax', 'irs', 'deduction', 'income', 'filing'],
            Domain.BUILDING: ['code', 'construction', 'building', 'occupancy', 'fire'],
            Domain.FINANCIAL: ['revenue', 'earnings', 'quarter', 'fiscal', 'investor'],
        }

        for domain, keywords in domain_keywords.items():
            domain_scores[domain] += sum(1 for kw in keywords if kw in text_lower)

        # Return highest scoring domain
        best_domain = max(domain_scores, key=domain_scores.get)
        if domain_scores[best_domain] > 0:
            return best_domain
        return Domain.GENERAL

    def save_to_file(self, filepath: str):
        """Save database to JSON file."""
        data = {}
        for key, definitions in self.terms.items():
            data[key] = [
                {
                    'term': d.term,
                    'expansion': d.expansion,
                    'domain': d.domain.value,
                    'definition': d.definition,
                    'aliases': d.aliases,
                    'related_terms': d.related_terms,
                    'context_keywords': d.context_keywords,
                }
                for d in definitions
            ]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str):
        """Load additional terms from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for key, definitions in data.items():
            for d in definitions:
                term_def = TermDefinition(
                    term=d['term'],
                    expansion=d['expansion'],
                    domain=Domain(d['domain']),
                    definition=d.get('definition', ''),
                    aliases=d.get('aliases', []),
                    related_terms=d.get('related_terms', []),
                    context_keywords=d.get('context_keywords', []),
                )
                self.add_term(term_def)


class QueryEnhancer:
    """
    Enhances queries using terminology database.

    Features:
    - Acronym expansion
    - Synonym addition
    - Domain-specific query reformulation
    """

    def __init__(self, acronym_db: AcronymDatabase = None):
        self.acronym_db = acronym_db or AcronymDatabase()

    def enhance(
        self,
        query: str,
        domain: Domain = None,
        expand_acronyms: bool = True,
        add_synonyms: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance a query for better retrieval.

        Returns:
            Dict with:
            - original_query
            - enhanced_query
            - expansions
            - detected_domain
            - search_terms (for BM25)
        """
        # Detect domain if not provided
        if domain is None:
            domain = self.acronym_db.detect_domain(query)

        enhanced = query
        expansions = []

        # Expand acronyms
        if expand_acronyms:
            enhanced, acr_expansions = self.acronym_db.expand_query(query, domain)
            expansions.extend(acr_expansions)

        # Collect search terms for BM25
        search_terms = set(query.lower().split())

        # Add related terms
        if add_synonyms:
            for word in query.split():
                related = self.acronym_db.get_related_terms(word, domain)
                search_terms.update(r.lower() for r in related)

        return {
            'original_query': query,
            'enhanced_query': enhanced,
            'expansions': expansions,
            'detected_domain': domain.value,
            'search_terms': list(search_terms)
        }


# Example usage
if __name__ == "__main__":
    db = AcronymDatabase()

    # Test disambiguation
    print("Testing IRC disambiguation:")

    # Tax context
    tax_context = "What is IRC Section 199A about qualified business income?"
    defn = db.get_term("IRC", context=tax_context)
    print(f"  Tax context: IRC = {defn.expansion}")

    # Building context
    building_context = "Does the IRC require smoke detectors in residential buildings?"
    defn = db.get_term("IRC", context=building_context)
    print(f"  Building context: IRC = {defn.expansion}")

    # Query expansion
    print("\nQuery expansion:")
    enhancer = QueryEnhancer(db)

    result = enhancer.enhance("What are the QSBS requirements under IRC Section 1202?")
    print(f"  Original: {result['original_query']}")
    print(f"  Enhanced: {result['enhanced_query']}")
    print(f"  Domain: {result['detected_domain']}")
    print(f"  Expansions: {result['expansions']}")
