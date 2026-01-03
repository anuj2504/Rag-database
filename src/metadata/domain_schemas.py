"""
Domain-Specific Metadata Schemas and Extraction.

Key insight from enterprise RAG:
- Metadata architecture has HIGHER ROI than embedding model upgrades
- Domain-specific schemas enable precise filtering
- Simple keyword matching >> LLM extraction (consistency matters)

Domains supported:
- Contracts & Legal Agreements
- IRC (Internal Revenue Code) & Tax regulations
- Building Codes (IBC, local codes)
- Rules & Regulations (CFR, state laws)
- Financial Reports (10-K, quarterly, budgets)
- Design Documents (specifications, drawings)
"""
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Base Schema Classes
# =============================================================================

@dataclass
class ExtractedMetadata:
    """Container for extracted metadata."""
    document_type: str
    document_subtype: Optional[str] = None
    confidence: float = 0.0

    # Common fields
    title: Optional[str] = None
    date: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None

    # Domain-specific fields (populated based on document type)
    domain_metadata: Dict[str, Any] = field(default_factory=dict)

    # Extracted entities
    parties: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)

    # For filtering
    filter_tags: Dict[str, Any] = field(default_factory=dict)


class MetadataExtractor(ABC):
    """Base class for domain-specific metadata extraction."""

    @abstractmethod
    def extract(self, text: str, filename: str = None) -> ExtractedMetadata:
        """Extract metadata from document text."""
        pass

    @abstractmethod
    def get_filter_schema(self) -> Dict[str, str]:
        """Return schema of available filter fields."""
        pass


# =============================================================================
# Contract & Legal Documents
# =============================================================================

class ContractType(Enum):
    """Types of contracts/legal documents."""
    SERVICE_AGREEMENT = "service_agreement"
    EMPLOYMENT = "employment"
    NDA = "nda"
    LEASE = "lease"
    LICENSE = "license"
    PURCHASE = "purchase"
    PARTNERSHIP = "partnership"
    AMENDMENT = "amendment"
    SETTLEMENT = "settlement"
    OTHER = "other"


class ContractMetadataExtractor(MetadataExtractor):
    """
    Metadata extraction for contracts and legal agreements.

    Uses keyword matching (NOT LLM) for consistency.
    """

    # Contract type detection keywords
    CONTRACT_TYPE_KEYWORDS = {
        ContractType.SERVICE_AGREEMENT: [
            'services agreement', 'consulting agreement', 'service contract',
            'professional services', 'master services'
        ],
        ContractType.EMPLOYMENT: [
            'employment agreement', 'employment contract', 'offer letter',
            'at-will employment', 'separation agreement'
        ],
        ContractType.NDA: [
            'non-disclosure', 'confidentiality agreement', 'nda',
            'confidential information', 'proprietary information'
        ],
        ContractType.LEASE: [
            'lease agreement', 'rental agreement', 'lease contract',
            'landlord', 'tenant', 'premises'
        ],
        ContractType.LICENSE: [
            'license agreement', 'software license', 'intellectual property',
            'licensor', 'licensee', 'royalty'
        ],
        ContractType.PURCHASE: [
            'purchase agreement', 'sale agreement', 'asset purchase',
            'stock purchase', 'acquisition agreement'
        ],
        ContractType.PARTNERSHIP: [
            'partnership agreement', 'joint venture', 'operating agreement',
            'llc agreement', 'shareholder agreement'
        ],
        ContractType.AMENDMENT: [
            'amendment', 'addendum', 'modification', 'supplement to'
        ],
        ContractType.SETTLEMENT: [
            'settlement agreement', 'release', 'mutual release',
            'dispute resolution'
        ],
    }

    # Key legal terms for extraction
    LEGAL_TERMS = [
        'indemnification', 'liability', 'termination', 'breach',
        'force majeure', 'governing law', 'jurisdiction', 'arbitration',
        'confidentiality', 'non-compete', 'warranty', 'representation',
        'assignment', 'waiver', 'notice', 'amendment'
    ]

    # Party extraction patterns
    PARTY_PATTERNS = [
        r'between\s+([A-Z][A-Za-z\s,\.]+(?:LLC|Inc|Corp|Corporation|Company|Ltd))',
        r'"([A-Z][A-Za-z\s]+)"\s*\(',
        r'(?:Party|Parties):\s*([A-Z][A-Za-z\s,]+)',
    ]

    def __init__(self):
        self.party_patterns = [re.compile(p, re.IGNORECASE) for p in self.PARTY_PATTERNS]

    def extract(self, text: str, filename: str = None) -> ExtractedMetadata:
        """Extract contract metadata."""
        text_lower = text.lower()

        # Detect contract type
        contract_type, confidence = self._detect_contract_type(text_lower)

        # Extract parties
        parties = self._extract_parties(text)

        # Extract dates
        dates = self._extract_dates(text)

        # Extract legal terms present
        present_terms = [term for term in self.LEGAL_TERMS if term in text_lower]

        # Extract monetary values
        monetary_values = self._extract_monetary_values(text)

        # Build filter tags for search
        filter_tags = {
            'contract_type': contract_type.value,
            'has_indemnification': 'indemnification' in text_lower,
            'has_termination_clause': 'termination' in text_lower,
            'has_confidentiality': 'confidential' in text_lower,
            'has_non_compete': 'non-compete' in text_lower or 'noncompete' in text_lower,
            'has_arbitration': 'arbitration' in text_lower,
            'governing_law': self._extract_governing_law(text),
        }

        return ExtractedMetadata(
            document_type='contract',
            document_subtype=contract_type.value,
            confidence=confidence,
            title=self._extract_title(text),
            effective_date=dates.get('effective'),
            expiration_date=dates.get('expiration'),
            domain_metadata={
                'contract_type': contract_type.value,
                'monetary_values': monetary_values,
                'legal_terms': present_terms,
            },
            parties=parties,
            key_terms=present_terms,
            filter_tags=filter_tags
        )

    def _detect_contract_type(self, text_lower: str) -> tuple:
        """Detect contract type using keyword matching."""
        best_type = ContractType.OTHER
        best_score = 0

        for ctype, keywords in self.CONTRACT_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_type = ctype

        confidence = min(1.0, best_score / 3)  # 3+ matches = high confidence
        return best_type, confidence

    def _extract_parties(self, text: str) -> List[str]:
        """Extract party names."""
        parties = []
        for pattern in self.party_patterns:
            matches = pattern.findall(text[:5000])  # Check first part
            parties.extend(matches)

        # Deduplicate and clean
        cleaned = list(set(p.strip() for p in parties if len(p.strip()) > 3))
        return cleaned[:10]  # Limit to 10 parties

    def _extract_dates(self, text: str) -> Dict[str, str]:
        """Extract key dates."""
        dates = {}

        # Effective date patterns
        effective_pattern = r'effective\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})'
        match = re.search(effective_pattern, text, re.IGNORECASE)
        if match:
            dates['effective'] = match.group(1)

        # Expiration/termination date
        expiry_pattern = r'(?:expires?|terminat(?:es?|ion))\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{4})'
        match = re.search(expiry_pattern, text, re.IGNORECASE)
        if match:
            dates['expiration'] = match.group(1)

        return dates

    def _extract_monetary_values(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary values mentioned."""
        values = []
        pattern = r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:per\s+(\w+)|(\w+))?'
        for match in re.finditer(pattern, text):
            amount = match.group(1).replace(',', '')
            period = match.group(2) or match.group(3)
            values.append({
                'amount': float(amount),
                'period': period,
                'raw': match.group(0)
            })
        return values[:20]  # Limit

    def _extract_governing_law(self, text: str) -> Optional[str]:
        """Extract governing law jurisdiction."""
        pattern = r'(?:governed by|governing law)[:\s]+(?:the\s+)?(?:laws?\s+of\s+)?(?:the\s+)?(?:State\s+of\s+)?(\w+(?:\s+\w+)?)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract document title."""
        lines = text.split('\n')[:20]
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                if re.match(r'^[A-Z][A-Z\s]+$', line):  # ALL CAPS title
                    return line
                if any(term in line.lower() for term in ['agreement', 'contract', 'amendment']):
                    return line
        return None

    def get_filter_schema(self) -> Dict[str, str]:
        """Return available filter fields."""
        return {
            'contract_type': 'string (service_agreement, employment, nda, lease, license, purchase, partnership, amendment, settlement, other)',
            'has_indemnification': 'boolean',
            'has_termination_clause': 'boolean',
            'has_confidentiality': 'boolean',
            'has_non_compete': 'boolean',
            'has_arbitration': 'boolean',
            'governing_law': 'string (state/jurisdiction)',
        }


# =============================================================================
# IRC & Tax Code Documents
# =============================================================================

class IRCMetadataExtractor(MetadataExtractor):
    """
    Metadata extraction for IRC (Internal Revenue Code) and tax documents.
    """

    # IRC section patterns
    IRC_SECTION_PATTERN = r'(?:IRC\s*)?(?:§|Section)\s*(\d+(?:\([a-z]\))?(?:\(\d+\))?)'

    # Tax form references
    TAX_FORM_PATTERN = r'Form\s+(\d+(?:-\w+)?)'

    # Tax categories
    TAX_CATEGORIES = {
        'income_tax': ['income tax', 'taxable income', 'gross income', 'adjusted gross'],
        'estate_tax': ['estate tax', 'gift tax', 'inheritance', 'decedent'],
        'employment_tax': ['payroll tax', 'fica', 'medicare', 'unemployment'],
        'excise_tax': ['excise tax', 'fuel tax', 'tobacco', 'alcohol'],
        'corporate_tax': ['corporate tax', 'c corporation', 's corporation'],
        'capital_gains': ['capital gain', 'capital loss', 'basis', 'depreciation'],
        'deductions': ['deduction', 'exemption', 'credit', 'charitable'],
    }

    def extract(self, text: str, filename: str = None) -> ExtractedMetadata:
        """Extract IRC/tax document metadata."""
        text_lower = text.lower()

        # Extract IRC sections referenced
        irc_sections = self._extract_irc_sections(text)

        # Extract tax forms referenced
        tax_forms = self._extract_tax_forms(text)

        # Determine tax categories
        categories = self._determine_categories(text_lower)

        # Extract tax years
        tax_years = self._extract_tax_years(text)

        # Build filter tags
        filter_tags = {
            'irc_sections': irc_sections,
            'tax_forms': tax_forms,
            'tax_categories': categories,
            'tax_years': tax_years,
            'is_regulation': 'regulation' in text_lower or 'treasury' in text_lower,
            'is_guidance': any(term in text_lower for term in ['notice', 'revenue ruling', 'revenue procedure']),
        }

        return ExtractedMetadata(
            document_type='irc_code',
            document_subtype=categories[0] if categories else 'general',
            confidence=0.8 if irc_sections else 0.5,
            domain_metadata={
                'irc_sections': irc_sections,
                'tax_forms': tax_forms,
                'categories': categories,
                'tax_years': tax_years,
            },
            references=irc_sections + tax_forms,
            key_terms=categories,
            filter_tags=filter_tags
        )

    def _extract_irc_sections(self, text: str) -> List[str]:
        """Extract IRC section references."""
        matches = re.findall(self.IRC_SECTION_PATTERN, text)
        return list(set(matches))[:50]

    def _extract_tax_forms(self, text: str) -> List[str]:
        """Extract tax form references."""
        matches = re.findall(self.TAX_FORM_PATTERN, text)
        return list(set(matches))[:20]

    def _determine_categories(self, text_lower: str) -> List[str]:
        """Determine applicable tax categories."""
        categories = []
        for category, keywords in self.TAX_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                categories.append(category)
        return categories

    def _extract_tax_years(self, text: str) -> List[str]:
        """Extract tax years mentioned."""
        current_year = datetime.now().year
        years = []
        for year in range(current_year - 20, current_year + 5):
            if str(year) in text:
                years.append(str(year))
        return years

    def get_filter_schema(self) -> Dict[str, str]:
        """Return available filter fields."""
        return {
            'irc_sections': 'list of strings (section numbers)',
            'tax_forms': 'list of strings (form numbers)',
            'tax_categories': 'list of strings (income_tax, estate_tax, etc.)',
            'tax_years': 'list of strings (years)',
            'is_regulation': 'boolean',
            'is_guidance': 'boolean',
        }


# =============================================================================
# Building Codes
# =============================================================================

class BuildingCodeMetadataExtractor(MetadataExtractor):
    """
    Metadata extraction for building codes (IBC, local codes).
    """

    # Code types
    CODE_TYPES = {
        'ibc': ['international building code', 'ibc'],
        'irc_building': ['international residential code'],  # Different from tax IRC
        'ifc': ['international fire code', 'ifc'],
        'imc': ['international mechanical code', 'imc'],
        'ipc': ['international plumbing code', 'ipc'],
        'iecc': ['international energy conservation code', 'iecc'],
        'nfpa': ['nfpa', 'national fire protection'],
        'ada': ['ada', 'americans with disabilities', 'accessibility'],
        'local': ['municipal code', 'city code', 'county code'],
    }

    # Building categories
    BUILDING_CATEGORIES = {
        'structural': ['structural', 'foundation', 'load-bearing', 'seismic'],
        'fire_safety': ['fire', 'sprinkler', 'egress', 'smoke', 'alarm'],
        'electrical': ['electrical', 'wiring', 'circuit', 'voltage'],
        'plumbing': ['plumbing', 'drainage', 'water supply', 'fixture'],
        'mechanical': ['hvac', 'ventilation', 'heating', 'cooling'],
        'accessibility': ['accessible', 'ada', 'wheelchair', 'ramp'],
        'energy': ['energy', 'insulation', 'efficiency', 'thermal'],
        'occupancy': ['occupancy', 'egress', 'exit', 'capacity'],
    }

    def extract(self, text: str, filename: str = None) -> ExtractedMetadata:
        """Extract building code metadata."""
        text_lower = text.lower()

        # Detect code type
        code_type = self._detect_code_type(text_lower)

        # Extract section references
        sections = self._extract_sections(text)

        # Determine categories
        categories = self._determine_categories(text_lower)

        # Extract code edition/year
        edition = self._extract_edition(text)

        # Build filter tags
        filter_tags = {
            'code_type': code_type,
            'sections': sections,
            'categories': categories,
            'edition': edition,
            'is_amendment': 'amendment' in text_lower or 'revision' in text_lower,
            'jurisdiction': self._extract_jurisdiction(text),
        }

        return ExtractedMetadata(
            document_type='building_code',
            document_subtype=code_type,
            confidence=0.8 if code_type != 'unknown' else 0.4,
            domain_metadata={
                'code_type': code_type,
                'sections': sections,
                'categories': categories,
                'edition': edition,
            },
            references=sections,
            key_terms=categories,
            filter_tags=filter_tags
        )

    def _detect_code_type(self, text_lower: str) -> str:
        """Detect which building code."""
        for code, keywords in self.CODE_TYPES.items():
            if any(kw in text_lower for kw in keywords):
                return code
        return 'unknown'

    def _extract_sections(self, text: str) -> List[str]:
        """Extract section references."""
        pattern = r'(?:Section|§)\s*(\d+(?:\.\d+)*)'
        matches = re.findall(pattern, text)
        return list(set(matches))[:50]

    def _determine_categories(self, text_lower: str) -> List[str]:
        """Determine applicable building categories."""
        categories = []
        for category, keywords in self.BUILDING_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                categories.append(category)
        return categories

    def _extract_edition(self, text: str) -> Optional[str]:
        """Extract code edition year."""
        pattern = r'(\d{4})\s*(?:edition|version|ibc|irc)'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _extract_jurisdiction(self, text: str) -> Optional[str]:
        """Extract jurisdiction if local code."""
        pattern = r'(?:City|County|State)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return None

    def get_filter_schema(self) -> Dict[str, str]:
        """Return available filter fields."""
        return {
            'code_type': 'string (ibc, irc_building, ifc, imc, ipc, iecc, nfpa, ada, local)',
            'sections': 'list of strings (section numbers)',
            'categories': 'list of strings (structural, fire_safety, etc.)',
            'edition': 'string (year)',
            'is_amendment': 'boolean',
            'jurisdiction': 'string',
        }


# =============================================================================
# Financial Reports
# =============================================================================

class FinancialMetadataExtractor(MetadataExtractor):
    """
    Metadata extraction for financial documents.
    """

    REPORT_TYPES = {
        '10-k': ['10-k', 'annual report', 'form 10-k'],
        '10-q': ['10-q', 'quarterly report', 'form 10-q'],
        '8-k': ['8-k', 'current report', 'form 8-k'],
        'earnings': ['earnings', 'earnings call', 'earnings release'],
        'budget': ['budget', 'forecast', 'projection'],
        'audit': ['audit', 'audited', 'auditor'],
        'investor': ['investor presentation', 'investor deck'],
    }

    FINANCIAL_METRICS = [
        'revenue', 'net income', 'ebitda', 'gross margin', 'operating income',
        'cash flow', 'assets', 'liabilities', 'equity', 'earnings per share',
        'return on equity', 'debt ratio', 'current ratio'
    ]

    def extract(self, text: str, filename: str = None) -> ExtractedMetadata:
        """Extract financial document metadata."""
        text_lower = text.lower()

        # Detect report type
        report_type = self._detect_report_type(text_lower)

        # Extract fiscal periods
        periods = self._extract_fiscal_periods(text)

        # Extract metrics mentioned
        metrics = [m for m in self.FINANCIAL_METRICS if m in text_lower]

        # Extract company name
        company = self._extract_company_name(text)

        # Extract monetary amounts
        amounts = self._extract_amounts(text)

        # Build filter tags
        filter_tags = {
            'report_type': report_type,
            'fiscal_periods': periods,
            'metrics': metrics,
            'company': company,
            'has_forward_looking': 'forward-looking' in text_lower or 'guidance' in text_lower,
            'has_risk_factors': 'risk factor' in text_lower,
        }

        return ExtractedMetadata(
            document_type='financial_report',
            document_subtype=report_type,
            confidence=0.8 if report_type != 'unknown' else 0.5,
            domain_metadata={
                'report_type': report_type,
                'fiscal_periods': periods,
                'metrics': metrics,
                'company': company,
                'key_amounts': amounts[:10],
            },
            parties=[company] if company else [],
            key_terms=metrics,
            filter_tags=filter_tags
        )

    def _detect_report_type(self, text_lower: str) -> str:
        """Detect financial report type."""
        for rtype, keywords in self.REPORT_TYPES.items():
            if any(kw in text_lower for kw in keywords):
                return rtype
        return 'unknown'

    def _extract_fiscal_periods(self, text: str) -> List[str]:
        """Extract fiscal periods."""
        periods = []

        # Q1 2023, FY2023, etc.
        pattern = r'(Q[1-4]\s*\d{4}|FY\s*\d{4}|\d{4}\s*Q[1-4])'
        matches = re.findall(pattern, text, re.IGNORECASE)
        periods.extend(matches)

        # Year ended December 31, 2023
        pattern = r'(?:year|quarter)\s+ended\s+(\w+\s+\d{1,2},?\s+\d{4})'
        matches = re.findall(pattern, text, re.IGNORECASE)
        periods.extend(matches)

        return list(set(periods))

    def _extract_company_name(self, text: str) -> Optional[str]:
        """Extract company name."""
        # Look for company patterns in first part of doc
        patterns = [
            r'^([A-Z][A-Za-z\s]+(?:Inc|Corp|Corporation|Company|LLC|Ltd))',
            r'(?:Form 10-[KQ]|Annual Report).{0,50}([A-Z][A-Za-z\s]+(?:Inc|Corp))',
        ]
        for pattern in patterns:
            match = re.search(pattern, text[:3000], re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    def _extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial amounts."""
        amounts = []
        # Match amounts with context
        pattern = r'(\w+(?:\s+\w+)?)\s*(?:of|was|were|is)?\s*\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            amounts.append({
                'context': match.group(1),
                'amount': match.group(2),
                'unit': match.group(3) or 'dollars'
            })
        return amounts

    def get_filter_schema(self) -> Dict[str, str]:
        """Return available filter fields."""
        return {
            'report_type': 'string (10-k, 10-q, 8-k, earnings, budget, audit, investor)',
            'fiscal_periods': 'list of strings',
            'metrics': 'list of strings',
            'company': 'string',
            'has_forward_looking': 'boolean',
            'has_risk_factors': 'boolean',
        }


# =============================================================================
# Unified Metadata Extractor
# =============================================================================

class UnifiedMetadataExtractor:
    """
    Routes documents to appropriate domain-specific extractor.

    This is the main entry point for metadata extraction.
    """

    def __init__(self):
        self.extractors = {
            'contract': ContractMetadataExtractor(),
            'irc_code': IRCMetadataExtractor(),
            'building_code': BuildingCodeMetadataExtractor(),
            'financial_report': FinancialMetadataExtractor(),
        }

        # Detection keywords for routing
        self.document_type_keywords = {
            'contract': ['agreement', 'contract', 'whereas', 'party', 'parties'],
            'irc_code': ['irc', 'internal revenue', 'tax code', 'treasury regulation'],
            'building_code': ['building code', 'ibc', 'fire code', 'occupancy'],
            'financial_report': ['10-k', '10-q', 'earnings', 'revenue', 'fiscal year'],
        }

    def extract(
        self,
        text: str,
        filename: str = None,
        document_type_hint: str = None
    ) -> ExtractedMetadata:
        """
        Extract metadata using appropriate domain extractor.

        Args:
            text: Document text
            filename: Optional filename for hints
            document_type_hint: Optional type hint to skip detection

        Returns:
            ExtractedMetadata with domain-specific fields
        """
        # Determine document type
        if document_type_hint and document_type_hint in self.extractors:
            doc_type = document_type_hint
        else:
            doc_type = self._detect_document_type(text, filename)

        # Extract using appropriate extractor
        if doc_type in self.extractors:
            return self.extractors[doc_type].extract(text, filename)
        else:
            # Fallback to contract extractor as default
            return self.extractors['contract'].extract(text, filename)

    def _detect_document_type(self, text: str, filename: str = None) -> str:
        """Detect document type from content."""
        text_lower = text.lower()

        # Check filename first
        if filename:
            filename_lower = filename.lower()
            if any(term in filename_lower for term in ['irc', 'tax', 'revenue']):
                return 'irc_code'
            if any(term in filename_lower for term in ['ibc', 'building', 'code']):
                return 'building_code'
            if any(term in filename_lower for term in ['10k', '10q', 'financial', 'earnings']):
                return 'financial_report'

        # Score each type
        scores = {}
        for doc_type, keywords in self.document_type_keywords.items():
            scores[doc_type] = sum(1 for kw in keywords if kw in text_lower)

        # Return highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)
            if scores[best_type] >= 2:
                return best_type

        return 'contract'  # Default

    def get_all_filter_schemas(self) -> Dict[str, Dict[str, str]]:
        """Return filter schemas for all document types."""
        return {
            doc_type: extractor.get_filter_schema()
            for doc_type, extractor in self.extractors.items()
        }


# Create __init__.py for package
def create_init():
    return """
from .domain_schemas import (
    ExtractedMetadata,
    ContractMetadataExtractor,
    IRCMetadataExtractor,
    BuildingCodeMetadataExtractor,
    FinancialMetadataExtractor,
    UnifiedMetadataExtractor,
)

__all__ = [
    'ExtractedMetadata',
    'ContractMetadataExtractor',
    'IRCMetadataExtractor',
    'BuildingCodeMetadataExtractor',
    'FinancialMetadataExtractor',
    'UnifiedMetadataExtractor',
]
"""


# Example usage
if __name__ == "__main__":
    extractor = UnifiedMetadataExtractor()

    # Test contract
    contract_text = """
    SERVICES AGREEMENT

    This Agreement is entered into as of January 1, 2024 between
    ABC Corporation ("Company") and XYZ Consulting LLC ("Consultant").

    WHEREAS, Company desires to engage Consultant...

    The governing law shall be the State of Delaware.
    """

    result = extractor.extract(contract_text)
    print(f"Type: {result.document_type}")
    print(f"Subtype: {result.document_subtype}")
    print(f"Parties: {result.parties}")
    print(f"Filter tags: {result.filter_tags}")
