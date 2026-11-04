"""Heuristic Rule-Based Transaction Classifier (Strictly Non-LLM).

Classifies bank statement transactions using deterministic pattern matching:
1. Priority-weighted regular expressions for Indian & Global banking narratives
2. Merchant/Brand keyword dictionaries (19 primary categories, 45+ sub-categories)
3. International payment rails (Wire Transfer, ACH, Direct Debit, SEPA, Zelle, Venmo, CashApp)
   and international merchants (Target, Costco, CVS, Walgreens, Tesco, Sainsbury)
4. Priority fix for Interest & Equity/Final/Interim Dividends above Payroll/Salary rules
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
import re
from typing import List, Optional, Tuple
from src.utils.models import SUPPORTED_CATEGORIES, TransactionRecord

logger = logging.getLogger(__name__)


@dataclass
class RuleMatchResult:
    category: str
    sub_category: str
    confidence: float
    rule_id: str
    matched_token: str


class HeuristicRuleClassifier:
    """Deterministic heuristic rule engine for bank transaction classification (19 Categories)."""

    CATEGORIES = list(SUPPORTED_CATEGORIES)

    def __init__(self):
        # Tuple format: (rule_id, category, sub_category, compiled_regex, polarity_filter, confidence)
        # polarity_filter: "CREDIT", "DEBIT", or "ANY"
        self.rules: List[Tuple[str, str, str, re.Pattern, str, float]] = [
            # 1. HIGHEST PRIORITY: Interest & Dividends (Moved above Salary so Equity/Final/Interim Dividends match first)
            (
                "RULE_INTEREST_DIVIDEND",
                "Salary & Income",
                "Interest & Dividends",
                re.compile(
                    r"\b(?:EQUITY\s+DIVIDEND|FINAL\s+DIVIDEND|INTERIM\s+DIVIDEND|SPECIAL\s+DIVIDEND|DIVIDEND\s+WARRANT|"
                    r"DIVIDEND|DIV\s+CR|ACH[-\s/]*CR.*DIV|INT\.?\s*PD|INTEREST\s+CREDIT|INTEREST\s+PAID|"
                    r"SB\s+INT|FD\s+INT|SAVINGS\s+INTEREST|RD\s+INTEREST)\b",
                    re.IGNORECASE,
                ),
                "CREDIT",
                0.98,
            ),
            # 2. Salary & Payroll (Credits)
            (
                "RULE_SALARY_PAYROLL",
                "Salary & Income",
                "Payroll / Salary",
                re.compile(
                    r"\b(?:SALARY|PAYROLL|STIPEND|WAGES|SAL\s+CR|EMP\s+PAY|MONTHLY\s+PAY|BONUS|"
                    r"DIRECT\s+DEPOSIT.*PAYROLL|ACH\s+CREDIT.*PAYROLL|BACS\s+SALARY|"
                    r"PAYMENT\s+RECEIVED.*PAYROLL|NORTHSHORE\s+SERVICES\s+PAYROLL)\b",
                    re.IGNORECASE,
                ),
                "CREDIT",
                0.98,
            ),
            (
                "RULE_FREELANCE_INVOICE",
                "Salary & Income",
                "Business / Consulting Income",
                re.compile(
                    r"\b(?:CONSULTING|FREELANCE|CLIENT\s+PMT|INVOICE|RETAINER|UPWORK|FIVERR|STRIPE\s+PAYOUT|"
                    r"RAZORPAY\s+SETTLEMENT|WIRE\s+TRANSFER.*CLIENT|PAYPAL\s+PAYOUT)\b",
                    re.IGNORECASE,
                ),
                "CREDIT",
                0.94,
            ),
            # 3. Refunds & Reversals (Credits)
            (
                "RULE_REFUND_REVERSAL_CR",
                "Shopping & E-Commerce",
                "Refund / Reversal",
                re.compile(r"\b(?:REFUND|REVERSAL|CASHBACK|REV[-\s/]|CHARGEBACK|MERCHANT\s+CREDIT)\b", re.IGNORECASE),
                "CREDIT",
                0.93,
            ),
            # 4. Government & Taxes (Phase 1 New Category)
            (
                "RULE_GOVT_INCOME_TAX",
                "Government & Taxes",
                "Income Tax",
                re.compile(
                    r"\b(?:TIN\s+NSDL|CBDT|INCOME\s+TAX|ADVANCE\s+TAX|SELF\s+ASSESSMENT\s+TAX|TDS\s+DEDUCTED|"
                    r"IRS\s+USATAXPYMT|IRS\s+TAX|HMRC\s+TAX|TAX\s+CHALLAN)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_GOVT_PROPERTY_TAX",
                "Government & Taxes",
                "Property Tax",
                re.compile(
                    r"\b(?:PROPERTY\s+TAX|MUNICIPAL\s+TAX|BBMP\s+TAX|BMC\s+PROPERTY|AMC\s+PROPERTY\s+TAX|COUNCIL\s+TAX)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_GOVT_SERVICES",
                "Government & Taxes",
                "Government Services",
                re.compile(
                    r"\b(?:MCA\s+GOV|MINISTRY\s+OF\s+CORPORATE|PASSPORT\s+SEVA|PASSPORT\s+FEE|CHALLAN|"
                    r"EPFO|PARIVAHAN|RTO\s+FEE|STAMP\s+DUTY|UIDAI|AADHAAR|DMV\s+FEE|USCIS\s+FEE)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            # 5. Education & Learning (Phase 1 New Category)
            (
                "RULE_EDUCATION_EDTECH",
                "Education & Learning",
                "EdTech",
                re.compile(
                    r"\b(?:BYJU'?S|BYJU|THINK\s+AND\s+LEARN|UNACADEMY|VEDANTU|UPGRAD|PHYSICSWALLAH|"
                    r"GREAT\s+LEARNING|SIMPLILEARN|SCALER\s+ACADEMY|WHITEHAT)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_EDUCATION_ONLINE_COURSES",
                "Education & Learning",
                "Online Courses",
                re.compile(
                    r"\b(?:COURSERA|UDEMY|EDX\.ORG|EDX\s+INC|KHAN\s+ACADEMY|PLURALSIGHT|LINKEDIN\s+LEARNING|"
                    r"DATACAMP|MASTERCLASS|SKILLSHARE|UDACITY)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_EDUCATION_SCHOOL_COLLEGE",
                "Education & Learning",
                "School/College Fees",
                re.compile(
                    r"\b(?:SCHOOL\s+FEE|COLLEGE\s+FEE|TUITION\s+FEE|TUITION\s+PMT|UNIVERSITY|SEMESTER\s+FEE|"
                    r"HOSTEL\s+FEE|IIT\s+FEE|IIM\s+FEE|EXAMINATION\s+FEE|ACADEMY\s+FEE)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            # 6. Charity & Donations (Phase 1 New Category)
            (
                "RULE_CHARITY_RELIGIOUS",
                "Charity & Donations",
                "Religious",
                re.compile(
                    r"\b(?:TEMPLE\s+TRUST|TIRUMALA\s+TIRUPATI|SHIRDI\s+SAI|ISKCON|TEMPLE|MOSQUE|MASJID|"
                    r"CHURCH\s+TITHE|CHURCH\s+OFFERING|CHURCH|GURUDWARA|DERASAR)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_CHARITY_CROWDFUNDING",
                "Charity & Donations",
                "Crowdfunding",
                re.compile(
                    r"\b(?:KETTO\s+ORG|KETTO|MILAAP\s+SOCIAL|MILAAP|IMPACT\s+GURU|GOFUNDME|KICKSTARTER\s+PLEDGE)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_CHARITY_NGO",
                "Charity & Donations",
                "NGO",
                re.compile(
                    r"\b(?:DONATION|CHARITY|NGO\s+CONTRIBUTION|NGO|GIVEINDIA|CRY\s+INDIA|GOONJ|AKSHAYA\s+PATRA|"
                    r"PM\s+CARES|RED\s+CROSS|UNICEF|DOCTORS\s+WITHOUT\s+BORDERS|WORLD\s+VISION|UNITED\s+WAY)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            # 7. Personal Care & Wellness (Phase 1 New Category)
            (
                "RULE_WELLNESS_GYM_FITNESS",
                "Personal Care & Wellness",
                "Gym",
                re.compile(
                    r"\b(?:CULT\s+FIT|CUREFIT|GOLD'?S\s+GYM|GOLD\s+GYM|ANYTIME\s+FITNESS|PLANET\s+FITNESS|"
                    r"EQUINOX\s+FITNESS|ORANGETHEORY|FITNESS\s+FIRST|GYM\s+MEMBERSHIP|GYM|FITNESS\s+STUDIO)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_WELLNESS_SALON",
                "Personal Care & Wellness",
                "Salon",
                re.compile(
                    r"\b(?:URBAN\s+COMPANY|URBANCLAP|LAKME\s+SALON|ENRICH\s+SALON|JAWED\s+HABIB|TONI\s*&\s*GUY|"
                    r"NATURALS\s+SALON|SUPERCUTS|GREAT\s+CLIPS|SALON|HAIR\s+STUDIO|BARBERSHOP)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_WELLNESS_SPA",
                "Personal Care & Wellness",
                "Wellness",
                re.compile(
                    r"\b(?:O2\s+SPA|TATTAVA\s+SPA|FOUR\s+FOUNTAINS\s+SPA|AYURVEDA\s+WELLNESS|SPA\s+TREATMENT|"
                    r"MASSAGE\s+ENY|WELLNESS\s+CENTER|SPA|YOGA\s+STUDIO|HEADSPACE|CALM\s+APP)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.95,
            ),
            # 8. Wallet & Recharge (Phase 1 New Category)
            (
                "RULE_WALLET_RECHARGE",
                "Wallet & Recharge",
                "Digital Wallet Top-ups",
                re.compile(
                    r"\b(?:PAYTM\s+WALLET|PHONEPE\s+WALLET|AMAZON\s+PAY\s+LOAD|AMAZON\s+WALLET\s+TOPUP|"
                    r"FREECHARGE\s+WALLET|FREECHARGE|MOBIKWIK\s+WALLET|MOBIKWIK|WALLET\s+LOAD|WALLET\s+TOPUP|"
                    r"APPLE\s+CASH\s+LOAD|PAYPAL\s+TOPUP)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            # 9. ATM & Cash Withdrawal
            (
                "RULE_ATM_CASH",
                "ATM & Cash Withdrawal",
                "ATM Cash Withdrawal",
                re.compile(
                    r"\b(?:ATM\s+WDL|ATM[-\s/]CASH|CASH\s+WDL|ATW[-\s/]|NFS[-\s/]CASH|SELF\s+CHQ\s+CASH|"
                    r"ATM\s+WITHDRAWAL|NON[-\s]CHASE\s+ATM|LINK\s+ATM\s+WDL)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.98,
            ),
            # 10. Loans, EMI & Credit Cards
            (
                "RULE_LOAN_EMI",
                "Loans, EMI & Credit Cards",
                "Loan EMI",
                re.compile(
                    r"\b(?:EMI\b|HOME\s+LOAN|AUTO\s+LOAN|PERSONAL\s+LOAN|BAJAJ\s+FINANCE|HDFC\s+LTD\s+LOAN|"
                    r"NACH[-\s/]DR.*LOAN|ACH[-\s/]DR.*FIN|DIRECT\s+DEBIT.*MORTGAGE|STUDENT\s+LOAN\s+PMT)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_CREDIT_CARD_BILL",
                "Loans, EMI & Credit Cards",
                "Credit Card Bill Payment",
                re.compile(
                    r"\b(?:CRED\s+CLUB|CRED\b|CC\s+BILL|CREDIT\s+CARD\s+PMT|BILLDESK.*CARD|SBI\s+CARD|"
                    r"ICICI\s+CARD|HDFC\s+CC|AMEX\s+EPAY|CHASE\s+CREDIT\s+CRD\s+AUTOPAY|CITI\s+CARD\s+ONLINE)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            # 11. Investments & Insurance
            (
                "RULE_MUTUAL_FUND_STOCKS",
                "Investments & Insurance",
                "Mutual Funds & Equities",
                re.compile(
                    r"\b(?:ZERODHA|GROWW|UPSTOX|ANGEL\s+ONE|KUVERA|COIN\s+SIP|MUTUAL\s+FUND|"
                    r"INDIAN\s+CLEARING\s+CORP|ICCL|NSE\s+CLEARING|SIP\s+DR|CAMS|KFINTECH|NIPPON\s+MF|"
                    r"SBI\s+MF|HDFC\s+MF|PPFAS|VANGUARD\s+BUY|FIDELITY\s+INV|ROBINHOOD\s+DEPOSIT|CHARLES\s+SCHWAB)\b",
                    re.IGNORECASE,
                ),
                "ANY",
                0.97,
            ),
            (
                "RULE_INSURANCE_PREMIUM",
                "Investments & Insurance",
                "Insurance Premium",
                re.compile(
                    r"\b(?:LIC\s+OF\s+INDIA|LIC\s+PREMIUM|HDFC\s+LIFE|ICICI\s+PRU|MAX\s+LIFE|STAR\s+HEALTH|"
                    r"TATA\s+AIA|POLICYBAZAAR|DIGIT\s+INSURANCE|ACKO|GEICO\s+INS|STATE\s+FARM\s+INS|AVIVA\s+INSURANCE|BUPA\s+HEALTH)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_FIXED_DEPOSIT_PPF",
                "Investments & Insurance",
                "Fixed Deposit / Provident Fund",
                re.compile(r"\b(?:FD\s+BOOKING|TERM\s+DEPOSIT|RD\s+INSTALLMENT|PPF\s+CONTRIBUTION|NPS\s+TRUST)\b", re.IGNORECASE),
                "DEBIT",
                0.95,
            ),
            # 12. Rent & Housing
            (
                "RULE_RENT_HOUSING",
                "Rent & Housing",
                "House Rent & Maintenance",
                re.compile(
                    r"\b(?:HOUSE\s+RENT|MONTHLY\s+RENT|APT\s+RENT|RENT\s+FOR|NOBROKER|MYGATE|"
                    r"ADDA\s+SOCIETY|SOCIETY\s+MAINT|FLAT\s+RENT|APARTMENT\s+LEASE|LANDLORD\s+RENT|"
                    r"SEPA\s+RENT|BACS\s+LETTINGS|AVALONBAY\s+COMMUNITIES|HARBOUR\s+PROPERTY|RAY\s+WHITE|"
                    r"LJ\s+HOOKER|MCGRATH|RENT\s+PAYMENT)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.95,
            ),
            # 13. Food & Dining
            (
                "RULE_FOOD_DELIVERY_DINING",
                "Food & Dining",
                "Restaurants & Food Delivery",
                re.compile(
                    r"\b(?:SWIGGY(?!\s+INSTAMART)|ZOMATO|DOMINOS|PIZZA\s+HUT|MCDONALDS|KFC|BURGER\s+KING|"
                    r"STARBUCKS|CAFE\s+COFFEE\s+DAY|SUBWAY|HALDIRAM|BARBEQUE\s+NATION|EATSURE|BASKIN|"
                    r"THEOBROMA|CHIPOTLE|DOORDASH|UBER\s+EATS|DELIVEROO|JUST\s+EAT|GRUBHUB|PANERA\s+BREAD|"
                    r"PRETAMANGER|PRET\s+A\s+MANGER|NANDO'?S|GUZMAN\s+Y\s+GOMEZ|GRILL'?D|BOOST\s+JUICE|"
                    r"CARD\s+PURCHASE\s*-\s*CAFE|RESTAURANT|CAFE|BAKERY|DINER|BISTRO)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            # 14. Groceries & Supermarkets (Includes Costco, Tesco, Sainsbury, Walmart, Whole Foods, Aldi, Lidl, Carrefour, Coles, Woolworths)
            (
                "RULE_GROCERIES_SUPERMARKET",
                "Groceries & Supermarkets",
                "Supermarket & Quick Commerce",
                re.compile(
                    r"\b(?:BLINKIT|ZEPTO|BIGBASKET|SWIGGY\s+INSTAMART|INSTAMART|DMART|AVENUE\s+SUPERMARTS|"
                    r"RELIANCE\s+FRESH|RELIANCE\s+SMART|MORE\s+RETAIL|SPENCERS|NATURE\s+BASKET|RATNADEEP|"
                    r"COUNTRY\s+DELIGHT|LULUHYPE|COSTCO\s+WHOLESALE|COSTCO|TESCO\s+STORES|TESCO\s+EXTRA|TESCO|"
                    r"SAINSBURY'?S|SAINSBURY|ASDA\s+SUPERSTORE|WAITROSE|ALDI\s+STORES|ALDI|LIDL|CARREFOUR|"
                    r"WALMART\s+SUPERCENTER|WALMART|WHOLE\s+FOODS|KROGER|TRADER\s+JOE'?S|SAFEWAY|PUBLIX|"
                    r"COLES\s+SUPERMARKET|COLES|WOOLWORTHS(?:\s+METRO)?|METRO\s*/\s*GROCERIES|IGA\s+SUPERMARKET|"
                    r"SUPERMARKET|GROCERY|GROCERIES)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            # 15. Healthcare & Medical (Includes CVS, Walgreens, Boots, Medicare)
            (
                "RULE_HEALTHCARE_PHARMACY",
                "Healthcare & Medical",
                "Hospitals, Labs & Pharmacy",
                re.compile(
                    r"\b(?:APOLLO\s+PHARMACY|APOLLO\s+HOSPITAL|PHARMEASY|TATA\s+1MG|1MG|NETMEDS|MEDPLUS|"
                    r"FORTIS|MAX\s+HEALTHCARE|MANIPAL\s+HOSPITAL|NARAYANA\s+HEALTH|DR\s+LAL\s+PATHLABS|"
                    r"METROPOLIS|THYROCARE|PRACTO|CVS\s+PHARMACY|CVS/PHARMACY|\bCVS\b|WALGREENS|BOOTS\s+PHARMACY|"
                    r"RITE\s+AID|QUEST\s+DIAGNOSTICS|LABCORP|CHEMIST\s+WAREHOUSE|PRICELINE\s+PHARMACY|PHARMACY\s+PURCHASE|"
                    r"HOSPITAL|CLINIC|DIAGNOSTIC|PHARMACY|MEDICAL\s+STORE|DENTAL)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_MEDICARE_REBATE",
                "Healthcare & Medical",
                "Hospitals, Labs & Pharmacy",
                re.compile(r"\b(?:MEDICARE\s+REBATE|HEALTH\s+REBATE|MEDICAL\s+REBATE)\b", re.IGNORECASE),
                "CREDIT",
                0.98,
            ),
            # 16. Shopping & E-Commerce (Includes Target, Best Buy, Home Depot, Lowe's, Bunnings)
            (
                "RULE_ECOMMERCE_RETAIL",
                "Shopping & E-Commerce",
                "Online & Retail Shopping",
                re.compile(
                    r"\b(?:AMAZON(?!\s+PRIME|\s+PAY\s+LOAD|\s+WALLET)|AMZN|FLIPKART|MYNTRA|AJIO|NYKAA|MEESHO|"
                    r"TATACLIQ|CROMA|RELIANCE\s+DIGITAL|VIJAY\s+SALES|DECATHLON|IKEA|ZARA|H&M|UNIQLO|"
                    r"PANTALOONS|WESTSIDE|SHOPPERS\s+STOP|LIFESTYLE|APPLE\s+STORE|TARGET\s+STORE|TARGET\s+T-\d+|\bTARGET\b|"
                    r"BEST\s+BUY|HOME\s+DEPOT|LOWE'?S|MACY'?S|NORDSTROM|JOHN\s+LEWIS|ARGOS|MARKS\s*&\s*SPENCER|EBAY|ETSY|"
                    r"BUNNINGS(?:\s+WAREHOUSE)?|KMART|BIG\s+W|OFFICEWORKS|HARVEY\s+NORMAN)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            # 17. Travel & Transport
            (
                "RULE_CAB_METRO_TRANSIT",
                "Travel & Transport",
                "Cab, Auto & Metro Transit",
                re.compile(
                    r"\b(?:UBER(?!\s+EATS)|OLA\s+CABS|OLACABS|RAPIDO|NAMMA\s+YATRI|BLUSMART|LYFT\s+RIDE|\bLYFT\b|"
                    r"DMRC|METRO\s+CARD|TFL\.GOV\.UK|OYSTER\s+CARD|MTA\s+NYCT|BART\s+CLIPPER|FASTAG|NHAI\s+TOLL|"
                    r"OPAL(?:\s+TOP\s*UP|\s+TRANSPORT)?|MYKI\s+PASS|MYKI|PAYTM\s+FASTAG|ICICI\s+FASTAG|E-ZPASS)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_FLIGHTS_TRAINS_HOTELS",
                "Travel & Transport",
                "Flights, Rail & Hotels",
                re.compile(
                    r"\b(?:IRCTC|MAKEMYTRIP|GOIBIBO|CLEARTRIP|YATRA|EASEMYTRIP|INDIGO|AIR\s+INDIA|VISTARA|"
                    r"AKASA|SPICEJET|DELTA\s+AIR|UNITED\s+AIRLINES|AMERICAN\s+AIRLINES|BRITISH\s+AIRWAYS|"
                    r"EMIRATES\s+AIR|LUFTHANSA|QANTAS|VIRGIN\s+AUSTRALIA|JETSTAR|AMTRAK|EUROSTAR|REDBUS|AIRBNB|OYO|TAJ\s+HOTELS|MARRIOTT|"
                    r"HILTON\s+HOTELS|HYATT|BOOKING\.COM|EXPEDIA|AGODA)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_FUEL_PETROL",
                "Travel & Transport",
                "Fuel & EV Charging",
                re.compile(
                    r"\b(?:INDIAN\s+OIL|IOCL|BHARAT\s+PETROLEUM|BPCL|HINDUSTAN\s+PETROLEUM|HPCL|SHELL\s+PETROL|"
                    r"SHELL\s+OIL|NAYARA\s+ENERGY|PETROL\s+PUMP|FUEL\s+STATION|FILLING\s+STATION|TATA\s+POWER\s+EZ|"
                    r"CARD\s+PURCHASE\s*-\s*FUEL|AMPOL|CALTEX|7-ELEVEN\s+FUEL|CHEVRON|EXXONMOBIL|BP\s+CONNECT|TEXACO|TESLA\s+SUPERCHARGER)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            # 18. Utilities & Bills
            (
                "RULE_TELECOM_INTERNET",
                "Utilities & Bills",
                "Mobile, Broadband & DTH",
                re.compile(
                    r"\b(?:JIO\s+FIBER|RELIANCE\s+JIO|JIO\s+PREPAID|AIRTEL|BHARTI\s+AIRTEL|VODAFONE|VI\s+PREPAID|"
                    r"ACT\s+FIBERNET|BSNL|TATA\s+PLAY|TATA\s+SKY|HATHWAY|SPECTRANET|VERIZON\s+WIRELESS|VERIZON|"
                    r"AT&T\s+MOBILITY|AT&T|T-MOBILE|COMCAST\s+XFINITY|XFINITY|BT\s+GROUP|VODAFONE\s+UK|SKY\s+DIGITAL|"
                    r"TELSTRA(?:\s+MOBILE)?|OPTUS|TPG\s+TELECOM)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.96,
            ),
            (
                "RULE_ELECTRICITY_GAS_WATER",
                "Utilities & Bills",
                "Electricity, Gas & Water",
                re.compile(
                    r"\b(?:BESCOM|TATA\s+POWER(?!\s+EZ)|ADANI\s+ELECTRICITY|TORRENT\s+POWER|MSEB|MAHADISCOM|"
                    r"BSES|CESC|UHBVN|IGL\s+GAS|MAHANAGAR\s+GAS|ADANI\s+GAS|INDANE|BHARAT\s+GAS|HP\s+GAS|"
                    r"WATER\s+BOARD|BWSSB|BBPS|CON\s+EDISON|PG&E\s+UTILITY|BRITISH\s+GAS|OCTOPUS\s+ENERGY|"
                    r"THAMES\s+WATER|EDF\s+ENERGY|ORIGIN\s+ENERGY|AGL\s+ENERGY|ENERGYAUSTRALIA|SYDNEY\s+WATER|"
                    r"ELECTRICITY\s+BILL|UTILITY\s+BILL)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            (
                "RULE_CLOUD_HOSTING",
                "Utilities & Bills",
                "Cloud & Hosting",
                re.compile(
                    r"\b(?:AWS(?:\s+CLOUD|\s+SERVICES)?|AMAZON\s+WEB\s+SERVICES|AZURE|GOOGLE\s+CLOUD|GCP|"
                    r"DIGITALOCEAN|LINODE|CLOUDFLARE|HEROKU|GITHUB|GITLAB|ORACLE\s+CLOUD)\b",
                    re.IGNORECASE,
                ),
                "ANY",
                0.97,
            ),
            # 19. Entertainment & Subscriptions
            (
                "RULE_STREAMING_ENTERTAINMENT",
                "Entertainment & Subscriptions",
                "OTT, Gaming & Movies",
                re.compile(
                    r"\b(?:NETFLIX|SPOTIFY|AMAZON\s+PRIME|PRIME\s+VIDEO|DISNEY\s+HOTSTAR|HOTSTAR|DISNEY\s+PLUS|"
                    r"YOUTUBE\s+PREMIUM|SONY\s+LIV|ZEE5|APPLE\s+MEDIA|HULU|HBO\s+MAX|BOOKMYSHOW|PVR\s+INOX|"
                    r"PVR\s+CINEMAS|AMC\s+THEATRES|REGAL\s+CINEMAS|ODEON\s+CINEMAS|STEAM\s+GAMES|PLAYSTATION|"
                    r"STAN\s+STREAMING|BINGE\s+STREAMING|KAYO\s+SPORTS|STREAMING\s+SUBSCRIPTION|"
                    r"XBOX|NINTENDO|OPENAI|CHATGPT|CLAUDE|GITHUB|ADOBE)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            # 20. Bank Fees, Taxes & Charges
            (
                "RULE_BANK_FEES_TAXES",
                "Bank Fees, Taxes & Charges",
                "Bank Charges & Taxes",
                re.compile(
                    r"\b(?:SMS\s+ALERT\s+(?:CHG|CHARGES)|MIN\s+BAL\s+(?:CHG|CHARGES)|PENALTY\s+CHARGES|AMB\s+SHORTFALL|DEBIT\s+CARD\s+AMC|"
                    r"ANNUAL\s+MAINT|CHEQUE\s+RETURN\s+CHG|ECS\s+RETURN\s+CHG|IMPS\s+CHG|NEFT\s+CHG|"
                    r"GST\s+@|CGST|SGST|SERVICE\s+CHARGE|OVERDRAFT\s+FEE|MONTHLY\s+MAINTENANCE\s+FEE|"
                    r"BANK\s+ACCOUNT\s+SERVICE\s+FEE|ACCOUNT\s+SERVICE\s+FEE|MONTHLY\s+SERVICE\s+FEE|"
                    r"WIRE\s+TRANSFER\s+FEE|FOREIGN\s+TRANSACTION\s+FEE|NSF\s+FEE)\b",
                    re.IGNORECASE,
                ),
                "DEBIT",
                0.97,
            ),
            # 21. Peer-to-Peer & International Transfers (Zelle, Venmo, CashApp, SEPA, Wire Transfer, Direct Debit, PayID, UPI P2P)
            (
                "RULE_P2P_INTERNATIONAL_TRANSFER",
                "Transfers & Peer-to-Peer",
                "Bank / UPI Transfer",
                re.compile(
                    r"\b(?:ZELLE\s+PAYMENT|ZELLE\s+TRANSFER|ZELLE\s+TO|ZELLE\s+FROM|VENMO\s+PAYMENT|VENMO\s+CASHOUT|"
                    r"VENMO|CASH\s*APP|CASHAPP|SEPA\s+TRANSFER|SEPA\s+CREDIT|FASTER\s+PAYMENTS|WIRE\s+TRANSFER|"
                    r"PAYID\s+TRANSFER|PAYID|OSKO|TRANSFER\s+TO\s+SAVINGS|NETBANK\s+TRANSFER|"
                    r"DIRECT\s+DEBIT\s+TRANSFER|SELF\s+TRANSFER|OWN\s+ACCOUNT|FAMILY\s+TRANSFER|TO\s+FRIEND|"
                    r"SPLIT\s+BILL|SPLIT\s+DINNER|SENT\s+FROM\s+PAYTM|"
                    r"UPI[/-](?:CR|DR)?[/-]?\d+[/-][A-Z\s]{3,25}[/-]\w+@(?:okaxis|okhdfcbank|okicici|oksbi|ybl|ibl|axl|paytm|apl))\b",
                    re.IGNORECASE,
                ),
                "ANY",
                0.88,
            ),
        ]

    def classify(self, txn: TransactionRecord) -> Optional[RuleMatchResult]:
        """Evaluates transaction against deterministic rules and returns match if found."""
        text = f"{txn.description} {txn.clean_description}".strip()
        polarity = "CREDIT" if txn.credit > 0 and txn.debit == 0 else "DEBIT"

        for rule_id, category, sub_cat, pattern, req_polarity, base_conf in self.rules:
            if req_polarity != "ANY" and req_polarity != polarity:
                continue
            m = pattern.search(text)
            if m:
                matched_tok = m.group(0).strip()
                conf = min(0.99, round(base_conf + (0.01 if len(matched_tok) >= 6 else 0.0), 3))
                logger.debug("Rule %s matched token '%s' -> %s (%s)", rule_id, matched_tok, category, sub_cat)
                return RuleMatchResult(
                    category=category,
                    sub_category=sub_cat,
                    confidence=conf,
                    rule_id=rule_id,
                    matched_token=matched_tok,
                )
        return None
