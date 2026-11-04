"""Externalized Training Corpus Generator (`data/training_corpus.csv`).

Generates a rich, diverse, realistic banking transaction corpus of 2,400+ labeled samples across:
- All 19 primary categories and 36 sub-categories
- 1,000+ International transaction patterns (US ACH, Wire Transfer, Direct Debit UK, SEPA EU,
  Zelle, Venmo, CashApp, Target, Costco, CVS, Walgreens, Tesco, Sainsbury, Aldi, Best Buy, etc.)
- 1,000+ P2P transfer variations (UPI, Zelle, Venmo, CashApp, SEPA, Faster Payments, IMPS, NEFT,
  personal names, split bills, family support, self transfers) and Indian domestic patterns
- OCR-corrupted & abbreviated edge cases (0/O, 1/I, 5/S substitutions, truncated POS narrations)
"""

from __future__ import annotations
import csv
import os
import random
from typing import List, Tuple


def build_training_corpus(output_csv_path: str, seed: int = 42) -> str:
    rng = random.Random(seed)
    rows: List[Tuple[str, str, float, str, str, str]] = []
    # Columns: description, polarity, amount, category, sub_category, region_tag

    # Template bank: (category, sub_category, polarity, region, amount_range, templates, merchants_or_entities)
    specs = [
        # 1. Salary & Income - Payroll / Salary
        (
            "Salary & Income",
            "Payroll / Salary",
            "CREDIT",
            "IN",
            (38000, 185000),
            [
                "NEFT CR {ifsc} {m} MONTHLY SALARY CREDIT {mon}",
                "ACH CR {m} PAYROLL WAGES FOR {mon}",
                "IMPS CR {ref} {m} EMP STIPEND AND BONUS",
                "NEFT/{ref}/{m}/SAL CR {mon} PAYROLL",
                "RTGS CR {m} EXECUTIVE REMUNERATION PAYROLL",
            ],
            [
                "INFOSYS LIMITED", "TCS LTD", "WIPRO TECHNOLOGIES", "HCL TECH LTD",
                "TECH MAHINDRA", "RELIANCE INDUSTRIES", "HDFC BANK PAYROLL",
                "ACCENTURE INDIA PVT LTD", "COGNIZANT TECHNOLOGY", "LARSEN AND TOUBRO",
                "FLIPKART INTERNET PAYROLL", "ZOMATO MEDIA SALARY",
            ],
        ),
        (
            "Salary & Income",
            "Payroll / Salary",
            "CREDIT",
            "INTL",
            (2800, 12500),
            [
                "DIRECT DEPOSIT {m} PAYROLL ACH CREDIT {ref}",
                "BACS CREDIT {m} MONTHLY SALARY WAGES",
                "SEPA CREDIT TRANSFER {m} GEHALT SALARY {mon}",
                "WIRE TRANSFER {m} BIWEEKLY PAYROLL DEPOSIT",
                "ACH CREDIT {m} DIRECT DEP PAYROLL",
            ],
            [
                "MICROSOFT CORP", "GOOGLE LLC", "AMAZON WEB SERVICES", "APPLE INC",
                "JP MORGAN CHASE PAYROLL", "DELOITTE LLP UK", "SIEMENS AG",
                "BARCLAYS SERVICES LTD", "META PLATFORMS INC", "STRIPE INC PAYROLL",
            ],
        ),
        # Salary & Income - Business / Consulting Income
        (
            "Salary & Income",
            "Business / Consulting Income",
            "CREDIT",
            "INTL",
            (800, 45000),
            [
                "WIRE TRANSFER {m} CLIENT CONSULTING INVOICE {ref}",
                "NEFT CR {m} FREELANCE RETAINER SETTLEMENT",
                "ACH CREDIT {m} MERCHANT PAYOUT SETTLEMENT",
                "SEPA CREDIT {m} PROFESSIONAL SERVICES INVOICE",
            ],
            [
                "UPWORK ESCROW INC", "FIVERR INTERNATIONAL", "RAZORPAY SOFTWARE SETTLEMENT",
                "STRIPE PAYMENTS UK", "PAYPAL PTE LTD PAYOUT", "TOPTAL LLC CONSULTING",
                "DEEL INC CONTRACTOR PMT", "PAYONEER INC CLIENT SETTLEMENT",
            ],
        ),
        # Salary & Income - Interest & Dividends
        (
            "Salary & Income",
            "Interest & Dividends",
            "CREDIT",
            "IN",
            (250, 18000),
            [
                "ACH CR {m} EQUITY DIVIDEND WARRANT {ref}",
                "NEFT CR {m} FINAL DIVIDEND FOR FY26",
                "CMS CR {m} INTERIM DIVIDEND PAYOUT",
                "INT.PD SAVINGS ACCOUNT QUARTERLY INTEREST CREDIT",
                "SB INT CREDIT FOR PERIOD {mon} TO DEC",
                "FD MATURITY INTEREST CREDIT TERM DEPOSIT {ref}",
            ],
            [
                "RELIANCE INDUSTRIES LTD", "TATA CONSULTANCY SERVICES", "HDFC BANK EQUITY",
                "INFOSYS LTD DIVIDEND", "ITC LIMITED DIVIDEND", "HINDUSTAN UNILEVER",
                "L&T EQUITY DIVIDEND", "COAL INDIA INTERIM DIV",
            ],
        ),
        (
            "Salary & Income",
            "Interest & Dividends",
            "CREDIT",
            "INTL",
            (45, 3200),
            [
                "ACH CREDIT {m} QUARTERLY EQUITY DIVIDEND",
                "DIRECT CREDIT {m} SAVINGS INTEREST PAID",
                "WIRE CREDIT {m} SPECIAL DIVIDEND DISTRIBUTION",
            ],
            [
                "VANGUARD BROKERAGE DIV", "FIDELITY INVESTMENTS DIVIDEND",
                "CHARLES SCHWAB INTEREST", "APPLE INC EQUITY DIVIDEND", "MARCUS BY GOLDMAN SACHS INT",
            ],
        ),

        # 2. Food & Dining
        (
            "Food & Dining",
            "Restaurants & Food Delivery",
            "DEBIT",
            "IN",
            (140, 2800),
            [
                "UPI/DR/{ref}/{m}/food@axisbank",
                "POS DEBIT VISA {m} DINING ORDER",
                "UPI-{ref}-{m}-ONLINE FOOD DELIVERY",
                "ECOM POS {m} RESTAURANT BILL",
            ],
            [
                "SWIGGY BUNDL TECHNOLOGIES", "ZOMATO LIMITED", "DOMINOS PIZZA JUBILANT",
                "STARBUCKS COFFEE TATA", "MCDONALDS CONNAUGHT PLAZA", "KFC SAPPHIRE FOODS",
                "BURGER KING INDIA", "SUBWAY SYSTEMS INDIA", "HALDIRAM SNACKS",
                "BARBEQUE NATION HOSPITALITY", "THEOBROMA BAKERY", "CAFE COFFEE DAY",
                "PIZZA HUT YUM RESTAURANTS", "BASKIN ROBBINS ICE CREAM", "EATSURE REBEL FOODS",
            ],
        ),
        (
            "Food & Dining",
            "Restaurants & Food Delivery",
            "DEBIT",
            "INTL",
            (12, 185),
            [
                "POS DEBIT {m} #{ref} CARD PURCHASE",
                "ACH DEBIT {m} ONLINE FOOD DELIVERY",
                "CONTACTLESS POS {m} RESTAURANT DINING",
                "VISA PURCHASE {m} FAST CASUAL DINER",
            ],
            [
                "DOORDASH*ORDER", "UBER EATS HELP.UBER.COM", "GRUBHUB*FOOD DELIVERY",
                "DELIVEROO.CO.UK ORDER", "JUST EAT UK", "CHIPOTLE MEXICAN GRILL",
                "PANERA BREAD CAFE", "PRET A MANGER LONDON", "NANDOS PERI PERI",
                "SHAKE SHACK BURGER", "CHICK-FIL-A #0412", "OLIVE GARDEN ITALIAN DINER",
            ],
        ),

        # 3. Groceries & Supermarkets
        (
            "Groceries & Supermarkets",
            "Supermarket & Quick Commerce",
            "DEBIT",
            "IN",
            (220, 6400),
            [
                "UPI/DR/{ref}/{m}/grocery@hdfcbank",
                "POS DEBIT {m} SUPERMARKET GROCERY",
                "UPI-{ref}-{m}-DAILY ESSENTIALS",
                "POS VISA {m} FRESH VEGETABLES AND DAIRY",
            ],
            [
                "BLINKIT GROFERS INDIA", "ZEPTO KIRANAKART TECH", "BIGBASKET SUPERMARKET",
                "SWIGGY INSTAMART GROCERY", "DMART AVENUE SUPERMARTS", "RELIANCE SMART BAZAAR",
                "RELIANCE FRESH SUPERMARKET", "MORE RETAIL HYPERMARKET", "RATNADEEP SUPERMARKET",
                "SPENCERS RETAIL GROCERY", "NATURE BASKET GOURMET", "COUNTRY DELIGHT DAIRY",
            ],
        ),
        (
            "Groceries & Supermarkets",
            "Supermarket & Quick Commerce",
            "DEBIT",
            "INTL",
            (35, 420),
            [
                "POS DEBIT {m} STORE #{ref} GROCERIES",
                "CONTACTLESS {m} SUPERMARKET PURCHASE",
                "DIRECT DEBIT {m} ONLINE GROCERY DELIVERY",
                "VISA CARD {m} WHOLESALE CLUB MARKET",
            ],
            [
                "COSTCO WHOLESALE #0481", "TESCO EXTRA SUPERSTORE", "TESCO STORES LTD",
                "SAINSBURYS SUPERMARKET", "WAITROSE AND PARTNERS", "ASDA SUPERSTORE UK",
                "WALMART SUPERCENTER", "WHOLE FOODS MARKET", "TRADER JOES #109",
                "KROGER FRESH GROCERY", "ALDI STORES SUPERMARKET", "LIDL GREAT BRITAIN",
                "CARREFOUR HYPERMARCHE", "SAFEWAY GROCERY STORE", "PUBLIX SUPER MARKETS",
            ],
        ),

        # 4. Shopping & E-Commerce
        (
            "Shopping & E-Commerce",
            "Online & Retail Shopping",
            "DEBIT",
            "IN",
            (499, 32000),
            [
                "UPI/DR/{ref}/{m}/shopping@icici",
                "POS ECOM {m} ONLINE RETAIL ORDER",
                "POS VISA {m} ELECTRONICS AND APPAREL",
                "UPI-{ref}-{m}-MARKETPLACE ORDER",
            ],
            [
                "AMAZON PAY INDIA ECOM", "FLIPKART INTERNET PVT LTD", "MYNTRA DESIGNS FASHION",
                "AJIO RELIANCE RETAIL", "NYKAA E RETAIL BEAUTY", "MEESHO FASHNEAR TECH",
                "TATACLIQ ECOMMERCE", "CROMA INFINITI RETAIL", "RELIANCE DIGITAL STORE",
                "VIJAY SALES ELECTRONICS", "DECATHLON SPORTS INDIA", "IKEA INDIA FURNISHING",
                "SHOPPERS STOP DEPARTMENT", "WESTSIDE TRENT RETAIL", "ZARA INDITEX APPAREL",
            ],
        ),
        (
            "Shopping & E-Commerce",
            "Online & Retail Shopping",
            "DEBIT",
            "INTL",
            (25, 1400),
            [
                "POS DEBIT {m} #{ref} RETAIL PURCHASE",
                "ECOM CARD PURCHASE {m} ONLINE ORDER",
                "ACH DEBIT {m} RETAIL STORE CHECKOUT",
                "VISA POS {m} DEPARTMENT STORE",
            ],
            [
                "TARGET STORE T-2184", "TARGET.COM *ORDER", "BEST BUY #0842 ELECTRONICS",
                "THE HOME DEPOT #4102", "LOWES HOME IMPROVEMENT", "AMZN MKTP US*ORDER",
                "AMAZON.CO.UK MARKETPLACE", "APPLE STORE #R042", "MACYS DEPARTMENT STORE",
                "NORDSTROM RETAIL", "JOHN LEWIS PARTNERSHIP", "MARKS AND SPENCER PLC",
                "ARGOS RETAIL UK", "EBAY COMMERCE INC", "ETSY.COM MARKETPLACE",
            ],
        ),
        (
            "Shopping & E-Commerce",
            "Refund / Reversal",
            "CREDIT",
            "INTL",
            (250, 6500),
            [
                "ECOM REFUND {m} REVERSAL FOR ORDER {ref}",
                "POS CREDIT {m} MERCHANT RETURN REFUND",
                "UPI CR {ref} {m} CASHBACK REVERSAL",
            ],
            [
                "AMAZON SELLER SERVICES", "FLIPKART INTERNET REFUND", "TARGET STORE RETURN",
                "MYNTRA DESIGNS REVERSAL", "BEST BUY MERCHANDISE REFUND", "COSTCO WHOLESALE REFUND",
            ],
        ),

        # 5. Utilities & Bills
        (
            "Utilities & Bills",
            "Electricity, Gas & Water",
            "DEBIT",
            "IN",
            (550, 5800),
            [
                "BBPS/DR/{ref}/{m} UTILITY BILL PMT",
                "UPI/DR/{ref}/{m}/bbps@axis",
                "NACH DR {m} MONTHLY UTILITY BILL",
            ],
            [
                "BESCOM ELECTRICITY BILL", "TORRENT POWER LIMITED", "ADANI ELECTRICITY MUMBAI",
                "TATA POWER DELHI DIST", "MAHADISCOM MSEB POWER", "BSES RAJDHANI POWER",
                "INDANE GAS LPG CYLINDER", "BHARAT GAS LPG REFILL", "MAHANAGAR GAS PIPED",
                "IGL INDRAPRASTHA GAS", "BWSSB WATER BOARD BILL",
            ],
        ),
        (
            "Utilities & Bills",
            "Electricity, Gas & Water",
            "DEBIT",
            "INTL",
            (65, 340),
            [
                "DIRECT DEBIT {m} MONTHLY ENERGY BILL {ref}",
                "ACH DEBIT {m} UTILITY AUTOPAY",
                "SEPA DIRECT DEBIT {m} ELECTRICITY GAS",
            ],
            [
                "CON EDISON ELECTRIC NY", "PG&E PACIFIC GAS ELECTRIC", "BRITISH GAS TRADING UK",
                "OCTOPUS ENERGY LTD", "THAMES WATER UTILITIES", "EDF ENERGY FRANCE",
                "DUKE ENERGY UTILITY", "NATIONAL GRID ELECTRIC",
            ],
        ),
        (
            "Utilities & Bills",
            "Mobile, Broadband & DTH",
            "DEBIT",
            "INTL",
            (399, 2400),
            [
                "UPI/DR/{ref}/{m} RECHARGE BILL",
                "ACH DEBIT {m} WIRELESS BROADBAND AUTOPAY",
                "DIRECT DEBIT {m} FIBER INTERNET BILL",
            ],
            [
                "JIO FIBER BROADBAND", "BHARTI AIRTEL MOBILE", "ACT FIBERNET INTERNET",
                "VODAFONE IDEA PREPAID", "TATA PLAY DTH RECHARGE", "VERIZON WIRELESS BILL",
                "AT&T MOBILITY AUTOPAY", "T-MOBILE USA WIRELESS", "COMCAST XFINITY CABLE",
                "BT GROUP BROADBAND UK", "VODAFONE UK LTD", "SKY DIGITAL TV UK",
            ],
        ),

        # 6. Rent & Housing
        (
            "Rent & Housing",
            "House Rent & Maintenance",
            "DEBIT",
            "IN",
            (3500, 48000),
            [
                "NEFT DR {ifsc} {m} MONTHLY RENT {mon}",
                "UPI/DR/{ref}/{m}/rent@hdfcbank",
                "IMPS DR {ref} {m} LEASE RENT PAYMENT",
            ],
            [
                "LANDLORD HOUSE RENT", "NOBROKER TECHNOLOGIES RENT", "MYGATE SOCIETY MAINT",
                "ADDA APARTMENT MAINTENANCE", "PRESTIGE ESTATES FLAT RENT",
                "BRIGADE GATEWAY RESIDENTIAL RENT", "MONTHLY APARTMENT LEASE RENT",
            ],
        ),
        (
            "Rent & Housing",
            "House Rent & Maintenance",
            "DEBIT",
            "INTL",
            (1100, 3600),
            [
                "ACH DEBIT {m} APARTMENT LEASE RENT {mon}",
                "SEPA DIRECT DEBIT {m} MONTHLY MIETE RENT",
                "BACS PAYMENT {m} RESIDENTIAL LETTINGS RENT",
                "WIRE TRANSFER {m} LANDLORD PROPERTY RENT",
            ],
            [
                "AVALONBAY COMMUNITIES RENT", "EQUITY RESIDENTIAL LEASE", "GREYSTAR PROPERTY RENT",
                "FOXTONS LETTINGS LONDON RENT", "VONOVIA SE MIETE RENT", "BILT REWARDS RENT PMT",
            ],
        ),

        # 7. Travel & Transport
        (
            "Travel & Transport",
            "Cab, Auto & Metro Transit",
            "DEBIT",
            "INTL",
            (120, 1500),
            [
                "UPI/DR/{ref}/{m} TRANSIT RIDE",
                "POS CONTACTLESS {m} COMMUTE FARE",
                "CARD PURCHASE {m} RIDE HAILING TRIP",
            ],
            [
                "UBER INDIA SYSTEMS CAB", "OLA CABS ANI TECH", "RAPIDO BIKE TAXI",
                "BLUSMART ELECTRIC CAB", "DMRC DELHI METRO CARD", "ICICI FASTAG TOLL",
                "PAYTM FASTAG NHAI TOLL", "LYFT *RIDE COMMUTE", "UBER *TRIP HELP.UBER.COM",
                "TFL.GOV.UK OYSTER TRAVEL", "MTA*NYCT SUBWAY FARE", "BART CLIPPER TRANSIT", "E-ZPASS TOLL NY",
            ],
        ),
        (
            "Travel & Transport",
            "Flights, Rail & Hotels",
            "DEBIT",
            "INTL",
            (1200, 38000),
            [
                "POS ECOM {m} BOOKING REF {ref}",
                "UPI/DR/{ref}/{m} TRAVEL TICKET",
                "VISA PURCHASE {m} RESERVATION",
            ],
            [
                "IRCTC RAIL TICKET BOOKING", "MAKEMYTRIP INDIA PVT LTD", "GOIBIBO FLIGHT HOTEL",
                "INDIGO INTERGLOBE AVIATION", "AIR INDIA EXPRESS", "VISTARA TATA SIA AIR",
                "DELTA AIR LINES TICKET", "UNITED AIRLINES HOUSTON", "BRITISH AIRWAYS PLC",
                "EMIRATES AIRLINE DUBAI", "LUFTHANSA GERMAN AIR", "AMTRAK RAIL PASSENGER",
                "EUROSTAR INTERNATIONAL", "AIRBNB *RESERVATION", "MARRIOTT BONVOY HOTELS",
                "HILTON HOTELS AND RESORTS", "TAJ HOTELS IHCL", "BOOKING.COM HOTEL RSV",
            ],
        ),
        (
            "Travel & Transport",
            "Fuel & EV Charging",
            "DEBIT",
            "INTL",
            (450, 4200),
            [
                "POS DEBIT {m} FUEL STATION #{ref}",
                "UPI/DR/{ref}/{m} PETROL DIESEL",
                "VISA PUMP {m} SERVICE STATION",
            ],
            [
                "INDIAN OIL IOCL PETROL PUMP", "BHARAT PETROLEUM BPCL FUEL",
                "HINDUSTAN PETROLEUM HPCL", "SHELL PETROL FILLING STATION",
                "NAYARA ENERGY FUEL", "TATA POWER EZ EV CHARGING",
                "SHELL OIL #49201", "CHEVRON SERVICE STATION", "EXXONMOBIL FUEL PUMP",
                "BP CONNECT SERVICE UK", "TESLA SUPERCHARGER US",
            ],
        ),

        # 8. Healthcare & Medical
        (
            "Healthcare & Medical",
            "Hospitals, Labs & Pharmacy",
            "DEBIT",
            "IN",
            (280, 14500),
            [
                "POS DEBIT {m} MEDICAL BILL #{ref}",
                "UPI/DR/{ref}/{m}/health@hdfcbank",
                "ECOM POS {m} DIAGNOSTIC AND PHARMACY",
            ],
            [
                "APOLLO PHARMACY MEDICINES", "APOLLO HOSPITALS ENTERPRISE",
                "TATA 1MG HEALTHCARE LAB", "PHARMEASY API HOLDINGS", "MEDPLUS CHEMIST DRUGGIST",
                "NETMEDS MARKETPLACE", "MANIPAL HOSPITALS OPD", "FORTIS HEALTHCARE HOSPITAL",
                "MAX HEALTHCARE CLINIC", "DR LAL PATHLABS DIAGNOSTIC", "METROPOLIS HEALTHCARE LAB",
                "THYROCARE TECHNOLOGIES", "NARAYANA HRUDAYALAYA",
            ],
        ),
        (
            "Healthcare & Medical",
            "Hospitals, Labs & Pharmacy",
            "DEBIT",
            "INTL",
            (18, 650),
            [
                "POS DEBIT {m} STORE #{ref} PHARMACY",
                "VISA CARD {m} MEDICAL PRESCRIPTION",
                "DIRECT DEBIT {m} CLINICAL DIAGNOSTICS",
            ],
            [
                "CVS/PHARMACY #04829", "CVS PHARMACY HEALTH", "WALGREENS DRUG STORE #1920",
                "BOOTS UK PHARMACY CHEMIST", "RITE AID PHARMACY", "QUEST DIAGNOSTICS LAB",
                "LABCORP MEDICAL TESTING", "KAISER PERMANENTE CLINIC", "BUPA DENTAL CARE UK",
            ],
        ),

        # 9. Entertainment & Subscriptions
        (
            "Entertainment & Subscriptions",
            "OTT, Gaming & Movies",
            "DEBIT",
            "INTL",
            (119, 2499),
            [
                "SI DR {m} MONTHLY SUBSCRIPTION {ref}",
                "UPI AUTOPAY {m} STREAMING MEMBERSHIP",
                "POS VISA {m} DIGITAL ENTERTAINMENT",
                "RECURRING CARD PMT {m} SUBSCRIPTION",
            ],
            [
                "NETFLIX.COM ENTERTAINMENT", "SPOTIFY INDIA MUSIC", "AMAZON PRIME MEMBERSHIP",
                "DISNEY HOTSTAR OTT", "YOUTUBE PREMIUM GOOGLE", "BOOKMYSHOW MOVIE TICKETS",
                "PVR INOX MULTIPLEX CINEMAS", "SONY LIV DIGITAL", "APPLE.COM/BILL ITUNES",
                "HULU LLC STREAMING", "HBO MAX SUBSCRIPTION", "AMC THEATRES BOX OFFICE",
                "ODEON CINEMAS UK", "STEAM GAMES VALVE CORP", "PLAYSTATION NETWORK SONY",
                "XBOX GAME PASS MICROSOFT", "OPENAI *CHATGPT SUBSCRIPTION", "ADOBE CREATIVE CLOUD",
            ],
        ),

        # 10. Investments & Insurance
        (
            "Investments & Insurance",
            "Mutual Funds & Equities",
            "DEBIT",
            "INTL",
            (2500, 65000),
            [
                "NACH DR {m} SYSTEMATIC INVESTMENT SIP {ref}",
                "UPI/DR/{ref}/{m}/invest@icici",
                "ACH DEBIT {m} BROKERAGE INVESTMENT DEPOSIT",
                "DIRECT DEBIT {m} EQUITY INDEX FUND",
            ],
            [
                "INDIAN CLEARING CORP ZERODHA COIN SIP", "GROWW NEXTBILLION MUTUAL FUND",
                "UPSTOX RKSV SECURITIES", "ANGEL ONE BROKING", "CAMS MUTUAL FUND SIP",
                "KFINTECH MF INVESTMENT", "SBI MUTUAL FUND SIP", "HDFC AMC MUTUAL FUND",
                "PPFAS PARAG PARIKH MF", "VANGUARD BUY INVESTMENT", "FIDELITY BROKERAGE DEPOSIT",
                "CHARLES SCHWAB INVESTMENT", "ROBINHOOD FUNDS TRANSFER", "HARGREAVES LANSDOWN ISA",
            ],
        ),
        (
            "Investments & Insurance",
            "Insurance Premium",
            "DEBIT",
            "INTL",
            (1200, 28000),
            [
                "NEFT DR {m} POLICY RENEWAL PREMIUM {ref}",
                "UPI/DR/{ref}/{m} INSURANCE PREMIUM",
                "ACH DEBIT {m} INSURANCE AUTOPAY",
                "DIRECT DEBIT {m} ANNUAL POLICY PREMIUM",
            ],
            [
                "LIC OF INDIA LIFE INSURANCE", "HDFC LIFE INSURANCE POLICY",
                "ICICI PRUDENTIAL LIFE INS", "MAX LIFE INSURANCE", "STAR HEALTH ALLIED INSURANCE",
                "TATA AIA LIFE INSURANCE", "POLICYBAZAAR INSURANCE", "DIGIT GENERAL INSURANCE",
                "ACKO GENERAL INSURANCE", "GEICO AUTO INSURANCE", "STATE FARM INSURANCE PMT",
                "AVIVA INSURANCE UK", "BUPA HEALTH INSURANCE",
            ],
        ),
        (
            "Investments & Insurance",
            "Fixed Deposit / Provident Fund",
            "DEBIT",
            "IN",
            (10000, 100000),
            [
                "FD BOOKING NEW TERM DEPOSIT ACCOUNT {ref}",
                "NEFT DR {m} CONTRIBUTION DEPOSIT",
            ],
            [
                "PPF PUBLIC PROVIDENT FUND", "NPS TRUST NATIONAL PENSION",
                "TERM FIXED DEPOSIT BOOKING", "RECURRING DEPOSIT RD INSTALLMENT",
            ],
        ),

        # 11. Loans, EMI & Credit Cards
        (
            "Loans, EMI & Credit Cards",
            "Loan EMI",
            "DEBIT",
            "INTL",
            (4200, 58000),
            [
                "NACH DR {m} MONTHLY EMI INSTALLMENT {ref}",
                "ACH DEBIT {m} LOAN REPAYMENT",
                "DIRECT DEBIT {m} MORTGAGE INSTALLMENT",
                "ECS DR {m} LOAN EMI",
            ],
            [
                "HDFC BANK HOME LOAN EMI", "SBI HOME LOAN EMI", "ICICI BANK AUTO LOAN EMI",
                "BAJAJ FINANCE CONSUMER EMI", "AXIS BANK PERSONAL LOAN EMI",
                "TATA CAPITAL HOUSING LOAN", "WELLS FARGO HOME MORTGAGE",
                "CHASE AUTO FINANCE LOAN", "NAVIENT STUDENT LOAN PMT", "HALIFAX MORTGAGE UK",
            ],
        ),
        (
            "Loans, EMI & Credit Cards",
            "Credit Card Bill Payment",
            "DEBIT",
            "INTL",
            (3500, 62000),
            [
                "UPI/DR/{ref}/{m} CREDIT CARD BILL PMT",
                "NEFT DR {m} CC STATEMENT PAYMENT {ref}",
                "ACH DEBIT {m} CREDIT CARD AUTOPAY",
                "DIRECT DEBIT {m} CARD SETTLEMENT",
            ],
            [
                "CRED CLUB CREDIT CARD BILL", "SBI CARDS AND PAYMENT SERVICES",
                "HDFC BANK CREDIT CARD BILLDESK", "ICICI BANK CREDIT CARD PMT",
                "AXIS BANK CC BILL PAYMENT", "AMERICAN EXPRESS AMEX EPAY",
                "CHASE CREDIT CRD AUTOPAY", "CITI CARD ONLINE PAYMENT",
                "CAPITAL ONE CARD AUTOPAY", "BARCLAYCARD PAYMENT UK",
            ],
        ),

        # 12. ATM & Cash Withdrawal
        (
            "ATM & Cash Withdrawal",
            "ATM Cash Withdrawal",
            "DEBIT",
            "INTL",
            (1000, 25000),
            [
                "ATM WDL NFS CASH WITHDRAWAL {m} {ref}",
                "ATW {m} ATM CASH DISPENSED",
                "NON-CHASE ATM WITHDRAWAL {m}",
                "LINK ATM CASH WITHDRAWAL {m}",
            ],
            [
                "HDFC ATM KORAMANGALA", "SBI ATM INDIRANAGAR", "ICICI ATM ANDHERI WEST",
                "AXIS BANK ATM CONNAUGHT PLACE", "SELF CHEQUE CASH WITHDRAWAL",
                "CHASE ATM #0921 NEW YORK", "BANK OF AMERICA ATM WDL", "BARCLAYS ATM LONDON",
            ],
        ),

        # 13. Bank Fees, Taxes & Charges
        (
            "Bank Fees, Taxes & Charges",
            "Bank Charges & Taxes",
            "DEBIT",
            "INTL",
            (15, 1200),
            [
                "{m} FOR PERIOD {mon} REF {ref}",
                "BANK CHARGE {m} INCLUDING GST",
                "FEE DEBIT {m} ACCOUNT SERVICE",
            ],
            [
                "SMS ALERT CHARGES FOR QTR", "DEBIT CARD ANNUAL MAINTENANCE FEE AMC",
                "MINIMUM AVERAGE BALANCE AMB SHORTFALL PENALTY", "ECS NACH RETURN BOUNCE CHARGES",
                "IMPS NEFT SERVICE CHARGE CGST SGST", "CHEQUE BOOK ISSUANCE CHARGES",
                "MONTHLY SERVICE MAINTENANCE FEE", "OVERDRAFT INSUFFICIENT FUNDS NSF FEE",
                "INTERNATIONAL WIRE TRANSFER FEE", "FOREIGN TRANSACTION CONVERSION FEE",
            ],
        ),

        # 14. NEW PHASE 1 CATEGORY: Education & Learning
        (
            "Education & Learning",
            "EdTech",
            "DEBIT",
            "IN",
            (1500, 42000),
            [
                "UPI/DR/{ref}/{m}/edtech@hdfcbank",
                "POS ECOM {m} COURSE SUBSCRIPTION",
                "NACH DR {m} EDTECH INSTALLMENT",
            ],
            [
                "BYJUS THINK AND LEARN PVT LTD", "UNACADEMY SORTING HAT TECH",
                "VEDANTU INNOVATIONS EDTECH", "UPGRAD EDUCATION ONLINE",
                "PHYSICSWALLAH EDTECH", "GREAT LEARNING EXECUTIVE", "SCALER ACADEMY TECH",
                "SIMPLILEARN SOLUTIONS",
            ],
        ),
        (
            "Education & Learning",
            "Online Courses",
            "DEBIT",
            "INTL",
            (499, 8500),
            [
                "POS VISA {m} ONLINE COURSE CERTIFICATION #{ref}",
                "UPI/DR/{ref}/{m} LEARNING SUBSCRIPTION",
                "ECOM CARD {m} PROFESSIONAL LEARNING",
            ],
            [
                "COURSERA.ORG ONLINE COURSE", "UDEMY INC ONLINE LEARNING",
                "EDX.ORG CERTIFICATE COURSE", "KHAN ACADEMY LEARNING",
                "PLURALSIGHT TECH SKILLS", "LINKEDIN LEARNING SUBSCRIPTION",
                "DATACAMP DATA SCIENCE", "MASTERCLASS ONLINE COURSE", "UDACITY NANODEGREE",
            ],
        ),
        (
            "Education & Learning",
            "School/College Fees",
            "DEBIT",
            "INTL",
            (12000, 145000),
            [
                "NEFT DR {ifsc} {m} SEMESTER FEE {ref}",
                "UPI/DR/{ref}/{m} ACADEMIC TUITION",
                "ACH DEBIT {m} UNIVERSITY TUITION PMT",
                "WIRE TRANSFER {m} COLLEGE TUITION FEE",
            ],
            [
                "DELHI PUBLIC SCHOOL FEE", "IIT BOMBAY SEMESTER TUITION FEE",
                "NATIONAL UNIVERSITY COLLEGE FEE", "BITS PILANI HOSTEL AND TUITION FEE",
                "IIM AHMEDABAD TERM FEE", "STANFORD UNIVERSITY TUITION PMT",
                "UNIVERSITY OF MANCHESTER TUITION FEE", "NEW YORK UNIVERSITY BURSAR TUITION",
            ],
        ),

        # 15. NEW PHASE 1 CATEGORY: Government & Taxes
        (
            "Government & Taxes",
            "Income Tax",
            "DEBIT",
            "INTL",
            (4500, 95000),
            [
                "NEFT DR {m} CHALLAN NO {ref}",
                "NETBANKING {m} TAX PAYMENT FY26",
                "ACH DEBIT {m} TAX SETTLEMENT",
            ],
            [
                "TIN NSDL INCOME TAX CHALLAN", "CBDT ADVANCE INCOME TAX PMT",
                "SELF ASSESSMENT INCOME TAX CBDT", "TDS DEDUCTED TRACES NSDL",
                "IRS USATAXPYMT FEDERAL INCOME TAX", "HMRC SELF ASSESSMENT TAX UK",
            ],
        ),
        (
            "Government & Taxes",
            "Property Tax",
            "DEBIT",
            "INTL",
            (2400, 28000),
            [
                "BBPS/DR/{ref}/{m} ANNUAL ASSESSMENT",
                "UPI/DR/{ref}/{m} MUNICIPAL RECEIPT",
                "DIRECT DEBIT {m} LOCAL AUTHORITY TAX",
            ],
            [
                "BBMP PROPERTY TAX BENGALURU", "BMC MUNICIPAL PROPERTY TAX MUMBAI",
                "AMC PROPERTY TAX AHMEDABAD", "GHMC MUNICIPAL PROPERTY TAX",
                "LONDON BOROUGH COUNCIL TAX", "COUNTY ASSESSOR PROPERTY TAX",
            ],
        ),
        (
            "Government & Taxes",
            "Government Services",
            "DEBIT",
            "INTL",
            (500, 15000),
            [
                "UPI/DR/{ref}/{m} GOVT PORTAL FEE",
                "NETBANKING {m} CHALLAN RECEIPT {ref}",
                "POS DEBIT {m} OFFICIAL SERVICE FEE",
            ],
            [
                "PASSPORT SEVA KENDRA APPLICATION FEE", "MCA MINISTRY OF CORPORATE AFFAIRS FILING",
                "EPFO EMPLOYER PROVIDENT CHALLAN", "PARIVAHAN RTO VEHICLE REGISTRATION CHALLAN",
                "E-STAMPING SHCIL STAMP DUTY", "CA DMV VEHICLE RENEWAL FEE", "USCIS APPLICATION FILING FEE",
            ],
        ),

        # 16. NEW PHASE 1 CATEGORY: Charity & Donations
        (
            "Charity & Donations",
            "Religious",
            "DEBIT",
            "IN",
            (251, 11000),
            [
                "UPI/DR/{ref}/{m} OFFERING DONATION",
                "NEFT DR {m} CHARITABLE TRUST RECEIPT {ref}",
            ],
            [
                "TIRUMALA TIRUPATI DEVASTHANAMS TEMPLE", "SHIRDI SAI BABA SANSTHAN TEMPLE",
                "ISKCON TEMPLE TRUST DONATION", "SIDDHIVINAYAK TEMPLE TRUST",
                "JAMA MASJID MOSQUE WELFARE TRUST", "ST MARYS CHURCH TITHE OFFERING",
                "GOLDEN TEMPLE GURUDWARA SEWA",
            ],
        ),
        (
            "Charity & Donations",
            "Crowdfunding",
            "DEBIT",
            "INTL",
            (500, 10000),
            [
                "UPI/DR/{ref}/{m} MEDICAL CAMPAIGN",
                "POS VISA {m} CROWDFUNDING CONTRIBUTION {ref}",
            ],
            [
                "KETTO ONLINE CROWDFUNDING", "MILAAP SOCIAL VENTURES CROWDFUNDING",
                "IMPACT GURU MEDICAL CAMPAIGN", "GOFUNDME*CHARITY CAMPAIGN",
            ],
        ),
        (
            "Charity & Donations",
            "NGO",
            "DEBIT",
            "INTL",
            (500, 25000),
            [
                "UPI/DR/{ref}/{m} 80G DONATION",
                "NEFT DR {m} CHARITY RELIEF FUND {ref}",
                "DIRECT DEBIT {m} MONTHLY PLEDGE DONATION",
            ],
            [
                "GIVEINDIA FOUNDATION NGO DONATION", "AKSHAYA PATRA FOUNDATION CHARITY",
                "CRY CHILD RIGHTS AND YOU NGO", "GOONJ CHARITABLE TRUST",
                "PM CARES RELIEF FUND DONATION", "AMERICAN RED CROSS CHARITY",
                "UNICEF MONTHLY CHILD DONATION", "DOCTORS WITHOUT BORDERS MSF NGO",
            ],
        ),

        # 17. NEW PHASE 1 CATEGORY: Personal Care & Wellness
        (
            "Personal Care & Wellness",
            "Gym",
            "DEBIT",
            "INTL",
            (999, 18000),
            [
                "UPI/DR/{ref}/{m} FITNESS MEMBERSHIP",
                "POS DEBIT {m} GYM SUBSCRIPTION #{ref}",
                "ACH DEBIT {m} MONTHLY CLUB DUES",
            ],
            [
                "CULT FIT HEALTH AND FITNESS", "CUREFIT GYM MEMBERSHIP",
                "GOLDS GYM FITNESS INSTITUTE", "ANYTIME FITNESS 24X7 CLUB",
                "FITNESS FIRST HEALTH CLUB", "PLANET FITNESS CLUB DUES",
                "EQUINOX FITNESS HOLDINGS", "ORANGETHEORY FITNESS STUDIO",
            ],
        ),
        (
            "Personal Care & Wellness",
            "Salon",
            "DEBIT",
            "INTL",
            (350, 4800),
            [
                "UPI/DR/{ref}/{m} GROOMING SERVICE",
                "POS VISA {m} HAIR AND BEAUTY #{ref}",
            ],
            [
                "URBAN COMPANY BEAUTY AND SALON", "LAKME LEVER BEAUTY SALON",
                "ENRICH HAIR AND SKIN SALON", "JAWED HABIB HAIR STUDIO",
                "TONI AND GUY ESSENSUALS SALON", "NATURALS UNISEX SALON",
                "SUPERCUTS HAIR SALON", "GREAT CLIPS BARBERSHOP",
            ],
        ),
        (
            "Personal Care & Wellness",
            "Wellness",
            "DEBIT",
            "INTL",
            (800, 6500),
            [
                "POS DEBIT {m} WELLNESS THERAPY #{ref}",
                "UPI/DR/{ref}/{m} REJUVENATION SPA",
            ],
            [
                "O2 SPA SALON AND WELLNESS", "FOUR FOUNTAINS DE-STRESS SPA",
                "TATTAVA WELLNESS SPA", "KERALA AYURVEDA WELLNESS",
                "MASSAGE ENVY SPA THERAPY", "HEADSPACE MINDFULNESS WELLNESS",
            ],
        ),

        # 18. NEW PHASE 1 CATEGORY: Wallet & Recharge
        (
            "Wallet & Recharge",
            "Digital Wallet Top-ups",
            "DEBIT",
            "INTL",
            (200, 10000),
            [
                "UPI/DR/{ref}/{m} ADD MONEY TOPUP",
                "POS DEBIT {m} PREPAID WALLET LOAD #{ref}",
            ],
            [
                "PAYTM WALLET LOAD ONE97", "PHONEPE WALLET TOPUP",
                "AMAZON PAY WALLET LOAD", "FREECHARGE PREPAID WALLET",
                "MOBIKWIK WALLET ADD MONEY", "APPLE CASH BALANCE LOAD",
                "PAYPAL WALLET BALANCE TOPUP",
            ],
        ),

        # 19. Australian & Asia-Pacific Specific Banking Patterns
        (
            "Utilities & Bills",
            "Electricity, Gas & Water",
            "DEBIT",
            "AU",
            (85, 450),
            [
                "{m} DIRECT DEBIT REF {ref}",
                "DIRECT DEBIT {m} BILL PAYMENT",
                "BPAY {m} ELECTRICITY & GAS ACC {ref}",
                "{m} UTILITY PAYMENT",
            ],
            [
                "ORIGIN ENERGY", "AGL ENERGY LTD", "ENERGYAUSTRALIA", "SYDNEY WATER",
                "YARRA VALLEY WATER", "ALINTA ENERGY", "RED ENERGY",
            ],
        ),
        (
            "Utilities & Bills",
            "Mobile, Broadband & DTH",
            "DEBIT",
            "AU",
            (45, 180),
            [
                "{m} MOBILE BILL REF {ref}",
                "DIRECT DEBIT {m} BROADBAND ACC {ref}",
                "{m} MONTHLY TELCO SERVICE",
                "CARD PURCHASE - {m}",
            ],
            [
                "TELSTRA", "OPTUS", "TPG TELECOM", "VODAFONE AUSTRALIA",
                "AUSSIE BROADBAND", "DODO SERVICES",
            ],
        ),
        (
            "Groceries & Supermarkets",
            "Supermarket & Quick Commerce",
            "DEBIT",
            "AU",
            (25, 320),
            [
                "{m} REF {ref}",
                "{m} SUPERMARKET SUPPLIES",
                "CARD PURCHASE - {m}",
                "POS DEBIT {m} STORE #{ref}",
            ],
            [
                "COLES SUPERMARKET", "WOOLWORTHS METRO / GROCERIES", "WOOLWORTHS SUPERMARKET",
                "ALDI STORES AUSTRALIA", "IGA SUPERMARKETS", "HARRIS FARM MARKETS",
            ],
        ),
        (
            "Shopping & E-Commerce",
            "Online & Retail Shopping",
            "DEBIT",
            "AU",
            (35, 750),
            [
                "{m} REF {ref}",
                "CARD PURCHASE - {m}",
                "POS DEBIT {m} #{ref}",
                "{m} RETAIL PURCHASE",
            ],
            [
                "BUNNINGS WAREHOUSE", "KMART AUSTRALIA", "TARGET AUSTRALIA",
                "BIG W DEPARTMENT STORE", "OFFICEWORKS", "HARVEY NORMAN",
                "JB HI-FI", "REBEL SPORT", "THE GOOD GUYS",
            ],
        ),
        (
            "Healthcare & Medical",
            "Hospitals, Labs & Pharmacy",
            "DEBIT",
            "AU",
            (20, 350),
            [
                "{m} PURCHASE REF {ref}",
                "CARD PURCHASE - {m}",
                "POS DEBIT {m}",
                "{m} PRESCRIPTION & OTC",
            ],
            [
                "CHEMIST WAREHOUSE", "PRICELINE PHARMACY", "TERRYWHITE CHEMMART",
                "PHARMACY PURCHASE", "BLOOMS THE CHEMIST", "SONIC HEALTHCARE LAB",
            ],
        ),
        (
            "Healthcare & Medical",
            "Hospitals, Labs & Pharmacy",
            "CREDIT",
            "AU",
            (35, 450),
            [
                "{m} REF {ref}",
                "{m} DIRECT BENEFIT DEPOSIT",
                "GOVT REBATE - {m}",
            ],
            [
                "MEDICARE REBATE", "MEDICARE BENEFIT REFUND", "PRIVATE HEALTH REBATE",
            ],
        ),
        (
            "Rent & Housing",
            "House Rent & Maintenance",
            "DEBIT",
            "AU",
            (350, 1850),
            [
                "RENT PAYMENT - {m}",
                "DIRECT DEBIT {m} RESIDENTIAL LEASE",
                "{m} FORTNIGHTLY RENT REF {ref}",
                "STRATA LEVY {m} BODY CORPORATE",
            ],
            [
                "HARBOUR PROPERTY", "RAY WHITE REAL ESTATE", "LJ HOOKER PROPERTY",
                "MCGRATH REALTY LEASE", "BELLE PROPERTY RENT", "FIRST NATIONAL REALTY",
            ],
        ),
        (
            "Travel & Transport",
            "Fuel & EV Charging",
            "DEBIT",
            "AU",
            (35, 140),
            [
                "CARD PURCHASE - FUEL STATION REF {ref}",
                "CARD PURCHASE - {m} FUEL",
                "POS DEBIT {m} SERVICE STATION",
                "{m} PETROL PUMP",
            ],
            [
                "BP AUSTRALIA", "AMPOL PETROL", "SHELL COLES EXPRESS",
                "7-ELEVEN FUEL", "UNITED PETROLEUM",
            ],
        ),
        (
            "Travel & Transport",
            "Cab, Auto & Metro Transit",
            "DEBIT",
            "AU",
            (10, 80),
            [
                "{m} TOP UP / TRANSPORT",
                "{m} TRANSIT AUTO LOAD REF {ref}",
                "CARD PURCHASE - {m}",
            ],
            [
                "OPAL", "MYKI", "GO CARD TRANSIT", "TRANSLINK TRANSIT",
            ],
        ),
        (
            "Food & Dining",
            "Restaurants & Food Delivery",
            "DEBIT",
            "AU",
            (12, 120),
            [
                "CARD PURCHASE - CAFE REF {ref}",
                "CARD PURCHASE - {m}",
                "POS DEBIT {m} DINING",
            ],
            [
                "GUZMAN Y GOMEZ", "BOOST JUICE", "GRILL'D BURGERS",
                "GLORIA JEAN'S COFFEES", "THE COFFEE CLUB",
            ],
        ),
        (
            "Entertainment & Subscriptions",
            "OTT, Gaming & Movies",
            "DEBIT",
            "AU",
            (10, 45),
            [
                "STREAMING SUBSCRIPTION REF {ref}",
                "{m} MONTHLY STREAMING",
                "CARD PURCHASE - {m}",
            ],
            [
                "STAN STREAMING", "BINGE ENTERTAINMENT", "KAYO SPORTS STREAMING",
                "OPTUS SPORT", "HOYTS CINEMAS", "EVENT CINEMAS",
            ],
        ),
        (
            "Bank Fees, Taxes & Charges",
            "Bank Charges & Taxes",
            "DEBIT",
            "AU",
            (5, 40),
            [
                "BANK ACCOUNT SERVICE FEE",
                "{m} MONTHLY CHARGE REF {ref}",
                "NON-AUD TRANSACTION FEE",
            ],
            [
                "ACCOUNT SERVICE FEE", "MONTHLY ACCOUNT FEE", "OVERDRAWN ACCOUNT FEE",
            ],
        ),
        (
            "Salary & Income",
            "Payroll / Salary",
            "CREDIT",
            "AU",
            (1800, 8500),
            [
                "PAYMENT RECEIVED - {m}",
                "DIRECT CREDIT {m} PAYROLL SALARY",
                "{m} WAGES & SALARY REF {ref}",
            ],
            [
                "NORTHSHORE SERVICES PAYROLL", "COMMONWEALTH PAYROLL CREDIT",
                "NSW HEALTH SALARY", "ATLASSIAN AUSTRALIA PAYROLL",
                "CANVA PTY LTD WAGES", "BHP GROUP SALARY DEPOSIT",
            ],
        ),
    ]

    ifscs = ["HDFC0000240", "ICIC0000104", "SBIN0004120", "UTIB0000152", "KKBK0000958", "PUNB0112000"]
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

    # Generate diverse merchant/category samples across specs (50 samples per spec)
    for cat, sub_cat, pol, region, (low, high), templates, entities in specs:
        for i in range(50):
            tmpl = templates[i % len(templates)]
            ent = entities[i % len(entities)]
            ref = str(rng.randint(10000000, 999999999))
            ifsc = rng.choice(ifscs)
            mon = rng.choice(months)
            amt = round(rng.uniform(low, high), 2)
            desc = tmpl.format(m=ent, ref=ref, ifsc=ifsc, mon=mon)

            # Inject realistic OCR corruption or abbreviation on ~15% of samples for robustness
            if i % 7 == 0:
                desc_ocr = (
                    desc.replace("0", "O")
                    .replace("I", "1", 1)
                    .replace("S", "5", 1)
                )
                rows.append((desc_ocr, pol, amt, cat, sub_cat, f"{region}_OCR"))
            else:
                rows.append((desc, pol, amt, cat, sub_cat, region))

    # Generate balanced P2P Transfer variations (Domestic UPI/IMPS/NEFT + International + Australian PayID/Osko)
    first_names = [
        "ROHIT", "PRIYA", "ANKIT", "SNEHA", "VIKRAM", "ANANYA", "RAHUL", "KAVITA",
        "ADITYA", "POOJA", "SIDDHARTH", "NEHA", "KARTHIK", "DIVYA", "ARJUN", "MEERA",
        "MICHAEL", "SARAH", "DAVID", "EMILY", "JAMES", "OLIVIA", "CARLOS", "SOPHIA",
        "LIAM", "EMMA", "NOAH", "AVA", "LUCAS", "MIA", "TAYLOR", "JORDAN", "MORGAN",
    ]
    last_names = [
        "SHARMA", "PATEL", "VERMA", "NAIR", "KULKARNI", "MEHTA", "IYER", "REDDY",
        "GUPTA", "JOSHI", "DESAI", "SINGH", "KUMAR", "CHATTERJEE", "MALHOTRA",
        "SMITH", "JOHNSON", "WILLIAMS", "BROWN", "JONES", "GARCIA", "MILLER", "DAVIS",
        "MUELLER", "SCHMIDT", "TAYLOR", "ANDERSON", "THOMAS", "WILSON", "MOORE",
    ]
    upi_handles = ["okicici", "okhdfcbank", "oksbi", "okaxis", "ybl", "ibl", "axl", "paytm", "apl"]
    p2p_memos = [
        "PERSONAL TRANSFER", "SPLIT DINNER BILL", "FAMILY SUPPORT", "SENT VIA PHONEPE",
        "SENT FROM PAYTM", "TO FRIEND", "SHARED CAB FARE", "WEEKEND TRIP SHARE",
        "GIFT TRANSFER", "REIMBURSEMENT", "SELF TRANSFER TO OWN ACCOUNT", "OWN SAVINGS TRANSFER",
        "COFFEE SPLIT", "LUNCH SHARE", "BIRTHDAY GIFT", "ROOMMATE SPLIT", "TRANSFER TO SAVINGS",
    ]

    p2p_templates_in = [
        ("UPI/DR/{ref}/{fn} {ln}/{fn_low}@{handle}/{memo}", "DEBIT"),
        ("UPI/CR/{ref}/{fn} {ln}/{fn_low}@{handle}/{memo}", "CREDIT"),
        ("IMPS-DR-{ref}-{fn} {ln}-{memo}", "DEBIT"),
        ("IMPS-CR-{ref}-{fn} {ln}-{memo}", "CREDIT"),
        ("NEFT DR {ifsc} {fn} {ln} {memo}", "DEBIT"),
    ]
    p2p_templates_intl = [
        ("ZELLE PAYMENT TO {fn} {ln} REF {ref} {memo}", "DEBIT"),
        ("ZELLE PAYMENT FROM {fn} {ln} REF {ref} {memo}", "CREDIT"),
        ("VENMO PAYMENT {fn} {ln} *{memo} #{ref}", "DEBIT"),
        ("VENMO CASHOUT FROM {fn} {ln} {ref}", "CREDIT"),
        ("CASH APP*{fn} {ln} P2P {memo} {ref}", "DEBIT"),
        ("SEPA TRANSFER TO {fn} {ln} IBAN DE89{ref} {memo}", "DEBIT"),
        ("FASTER PAYMENTS TRANSFER TO {fn} {ln} {memo}", "DEBIT"),
        ("WIRE TRANSFER TO {fn} {ln} {memo} REF {ref}", "DEBIT"),
    ]
    p2p_templates_au = [
        ("PAYID TRANSFER RECEIVED REF {ref}", "CREDIT"),
        ("PAYID TRANSFER TO {fn} {ln} REF {ref}", "DEBIT"),
        ("OSKO PAYMENT TO {fn} {ln} {memo} REF {ref}", "DEBIT"),
        ("OSKO PAYMENT FROM {fn} {ln} REF {ref}", "CREDIT"),
        ("TRANSFER TO SAVINGS REF {ref}", "DEBIT"),
        ("TRANSFER TO SAVINGS", "DEBIT"),
        ("NETBANK TRANSFER TO ACC {ref} {memo}", "DEBIT"),
    ]

    for i in range(120):
        fn = rng.choice(first_names[:16])
        ln = rng.choice(last_names[:15])
        handle = rng.choice(upi_handles)
        memo = rng.choice(p2p_memos)
        ref = str(rng.randint(100000000, 9999999999))
        ifsc = rng.choice(ifscs)
        tmpl, pol = p2p_templates_in[i % len(p2p_templates_in)]
        amt = round(rng.uniform(150.0, 25000.0), 2)
        desc = tmpl.format(ref=ref, fn=fn, ln=ln, fn_low=fn.lower(), handle=handle, memo=memo, ifsc=ifsc)
        rows.append((desc, pol, amt, "Transfers & Peer-to-Peer", "Bank / UPI Transfer", "IN"))

    for i in range(120):
        fn = rng.choice(first_names[16:])
        ln = rng.choice(last_names[15:])
        memo = rng.choice(p2p_memos)
        ref = str(rng.randint(100000000, 9999999999))
        tmpl, pol = p2p_templates_intl[i % len(p2p_templates_intl)]
        amt = round(rng.uniform(15.0, 2500.0), 2)
        desc = tmpl.format(ref=ref, fn=fn, ln=ln, memo=memo)
        rows.append((desc, pol, amt, "Transfers & Peer-to-Peer", "Bank / UPI Transfer", "INTL"))

    for i in range(120):
        fn = rng.choice(first_names[16:])
        ln = rng.choice(last_names[15:])
        memo = rng.choice(p2p_memos)
        ref = str(rng.randint(100000, 999999))
        tmpl, pol = p2p_templates_au[i % len(p2p_templates_au)]
        amt = round(rng.uniform(25.0, 3500.0), 2)
        desc = tmpl.format(ref=ref, fn=fn, ln=ln, memo=memo)
        rows.append((desc, pol, amt, "Transfers & Peer-to-Peer", "Bank / UPI Transfer", "AU"))

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["description", "polarity", "amount", "category", "sub_category", "region"])
        for r in rows:
            writer.writerow(r)

    return output_csv_path


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.path.join(base, "data", "training_corpus.csv")
    build_training_corpus(out)
    print(f"Generated {out}")
