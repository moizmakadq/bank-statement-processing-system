"""Realistic Multi-Bank Sample PDF Statement Generator (Text-Based & Scanned Image-Based).

Generates 3 benchmark PDF bank statements for testing and demonstration:
1. `HDFC_Bank_Text_Statement_Oct2026.pdf` (Digital Text-Based PDF, Dual Debit/Credit columns, 20 transactions)
2. `ICICI_Bank_Text_Statement_Oct2026.pdf` (Digital Text-Based PDF, Single Amount column with Dr/Cr indicators, 14 transactions including duplicate & refund edge cases)
3. `SBI_Scanned_Image_Statement_Oct2026.pdf` (True Scanned Image-Based PDF with 0 text layer, 1.2° simulated scanner skew & light paper grain, 12 transactions for testing OpenCV deskewing + Tesseract OCR)
"""

from __future__ import annotations
import os
import shutil
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _get_fallback_pregenerated_pdf(output_path: str) -> Optional[str]:
    """Helper to return or copy pregenerated benchmark PDF if available."""
    if os.path.exists(output_path):
        return output_path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_pdf = os.path.join(base_dir, "sample_data", "pdfs", os.path.basename(output_path))
    if os.path.exists(source_pdf) and os.path.abspath(source_pdf) != os.path.abspath(output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        shutil.copy2(source_pdf, output_path)
        return output_path
    return None


def generate_hdfc_text_pdf(output_path: str) -> str:
    """Generates a multi-column text-based HDFC Bank statement PDF."""
    if not HAS_REPORTLAB:
        fallback = _get_fallback_pregenerated_pdf(output_path)
        if fallback:
            return fallback
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1B365D"))
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=9, leading=13)

    story = []
    story.append(Paragraph("<b>HDFC BANK LIMITED - STATEMENT OF ACCOUNT</b>", title_style))
    story.append(Spacer(1, 4 * mm))

    meta_lines = [
        "<b>Account Holder Name:</b> AARAV MEHTA &nbsp;&nbsp;&nbsp;&nbsp; <b>Customer ID:</b> 84920144",
        "<b>Account Number:</b> 50100482910342 &nbsp;&nbsp;&nbsp;&nbsp; <b>Account Type:</b> Savings Max Account",
        "<b>IFSC Code:</b> HDFC0000240 &nbsp;&nbsp;&nbsp;&nbsp; <b>MICR Code:</b> 380240002 &nbsp;&nbsp;&nbsp;&nbsp; <b>Branch:</b> Navrangpura Ahmedabad",
        "<b>Statement Period:</b> 01/10/2026 to 31/10/2026 &nbsp;&nbsp;&nbsp;&nbsp; <b>Currency:</b> INR",
        "<b>Opening Balance:</b> 1,45,000.00 &nbsp;&nbsp;&nbsp;&nbsp; <b>Closing Balance:</b> 1,22,440.30",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, meta_style))
    story.append(Spacer(1, 5 * mm))

    # Transactions: (Date, Value Date, Narration, Ref No, Debit, Credit)
    raw_txns = [
        ("01/10/2026", "01/10/2026", "NEFT-CR-HDFC0000240-INFOSYS LIMITED-SALARY FOR SEP 2026", "NEFT6274819201", 0.0, 95000.00),
        ("02/10/2026", "02/10/2026", "UPI/DR/6275102938/LANDLORD HOUSE RENT OCT/owner@hdfcbank", "UPI6275102938", 26000.00, 0.0),
        ("03/10/2026", "03/10/2026", "UPI/DR/6276839201/BLINKIT GROFERS INDIA/blinkit@axisbank", "UPI6276839201", 1240.00, 0.0),
        ("05/10/2026", "05/10/2026", "NACH-DR-HDFC HOME LOAN EMI OCT2026", "NACH901827361", 32500.00, 0.0),
        ("06/10/2026", "06/10/2026", "UPI/DR/6279102931/SWIGGY FOOD ORDER/swiggy@icici", "UPI6279102931", 580.00, 0.0),
        ("07/10/2026", "07/10/2026", "NACH-DR-INDIAN CLEARING CORP ZERODHA COIN SIP", "NACH882716253", 15000.00, 0.0),
        ("09/10/2026", "09/10/2026", "BBPS/DR/TORRENT POWER LIMITED ELECTRICITY BILL", "BBPS773829102", 2890.00, 0.0),
        ("10/10/2026", "10/10/2026", "UPI/DR/6283910294/JIO FIBER BROADBAND RECHARGE/jio@paytm", "UPI6283910294", 999.00, 0.0),
        ("12/10/2026", "12/10/2026", "POS 416021XXXXXX8821 INDIAN OIL PETROL PUMP FUEL", "POS991827364", 2400.00, 0.0),
        ("14/10/2026", "14/10/2026", "POS 416021XXXXXX8821 AMAZON PAY INDIA ECOM SHOPPING", "POS992837465", 4599.00, 0.0),
        ("15/10/2026", "15/10/2026", "ATM-WDL-NFS CASH WITHDRAWAL CG ROAD AHMEDABAD", "ATM382910475", 10000.00, 0.0),
        ("17/10/2026", "17/10/2026", "UPI/DR/6290192837/UBER INDIA SYSTEMS CAB RIDE/uber@axis", "UPI6290192837", 420.00, 0.0),
        ("19/10/2026", "19/10/2026", "POS 416021XXXXXX8821 APOLLO PHARMACY MEDICAL STORE", "POS994827162", 1150.00, 0.0),
        ("21/10/2026", "21/10/2026", "SI-DR-NETFLIX ENTERTAINMENT MONTHLY SUBSCRIPTION", "SI772819203", 649.00, 0.0),
        ("23/10/2026", "23/10/2026", "UPI/DR/6296182736/CRED CLUB CREDIT CARD BILL PMT", "UPI6296182736", 18400.00, 0.0),
        ("25/10/2026", "25/10/2026", "ACH-CR-RELIANCE INDUSTRIES EQUITY DIVIDEND CREDIT", "ACH554829102", 0.0, 3250.00),
        ("27/10/2026", "27/10/2026", "POS 416021XXXXXX8821 DMART AVENUE SUPERMARTS GROCERY", "POS998172635", 3620.00, 0.0),
        ("28/10/2026", "28/10/2026", "UPI/DR/6301928374/ROHIT SHARMA FRIEND TRANSFER/rohit@okicici", "UPI6301928374", 2000.00, 0.0),
        ("30/10/2026", "30/10/2026", "SMS ALERT CHARGES FOR QTR ENDED OCT GST INCLUDED", "CHG0019283", 17.70, 0.0),
        ("31/10/2026", "31/10/2026", "INT.PD SAVINGS ACCOUNT QUARTERLY INTEREST CREDIT", "INT20261031", 0.0, 1655.00),
    ]

    table_data = [["Date", "Value Date", "Description", "Ref / Chq No", "Debit", "Credit", "Balance"]]
    bal = 145000.00
    for dt, vdt, desc, ref, dr, cr in raw_txns:
        bal = round(bal + cr - dr, 2)
        table_data.append(
            [
                dt,
                vdt,
                desc,
                ref,
                f"{dr:,.2f}" if dr > 0 else "",
                f"{cr:,.2f}" if cr > 0 else "",
                f"{bal:,.2f}",
            ]
        )

    col_widths = [20 * mm, 20 * mm, 64 * mm, 25 * mm, 18 * mm, 18 * mm, 21 * mm]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B365D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                ("ALIGN", (4, 1), (6, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    story.append(t)
    doc.build(story)
    return output_path


def generate_icici_text_pdf(output_path: str) -> str:
    """Generates an ICICI Bank statement PDF using a single Amount column with Cr/Dr indicators and edge cases."""
    if not HAS_REPORTLAB:
        fallback = _get_fallback_pregenerated_pdf(output_path)
        if fallback:
            return fallback
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#8B1E0F"))
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=9, leading=13)

    story = []
    story.append(Paragraph("<b>ICICI BANK LTD - DETAILED ACCOUNT STATEMENT</b>", title_style))
    story.append(Spacer(1, 4 * mm))

    meta_lines = [
        "<b>Account Holder Name:</b> ROHAN KULKARNI &nbsp;&nbsp;&nbsp;&nbsp; <b>Customer ID:</b> 55190283",
        "<b>Account Number:</b> 010401582910 &nbsp;&nbsp;&nbsp;&nbsp; <b>Account Type:</b> Privilege Savings",
        "<b>IFSC Code:</b> ICIC0000104 &nbsp;&nbsp;&nbsp;&nbsp; <b>MICR Code:</b> 400229015 &nbsp;&nbsp;&nbsp;&nbsp; <b>Branch:</b> Andheri West Mumbai",
        "<b>Statement Period:</b> 01/10/2026 to 31/10/2026 &nbsp;&nbsp;&nbsp;&nbsp; <b>Currency:</b> INR",
        "<b>Opening Balance:</b> 82,000.00 &nbsp;&nbsp;&nbsp;&nbsp; <b>Closing Balance:</b> 1,18,103.00",
    ]
    for line in meta_lines:
        story.append(Paragraph(line, meta_style))
    story.append(Spacer(1, 5 * mm))

    # Includes duplicate charge edge case on 08/10/2026 and ECOM Refund on 16/10/2026
    raw_txns = [
        ("01/10/2026", "NEFT-CR-ICIC0000104-TCS LTD MONTHLY PAYROLL SALARY", "NEFT8192011", 86000.00, "CR"),
        ("03/10/2026", "UPI/DR/71029384/NOBROKER TECHNOLOGIES HOUSE RENT", "UPI71029384", 24000.00, "DR"),
        ("05/10/2026", "UPI/DR/71048291/ZOMATO LIMITED ONLINE FOOD ORDER", "UPI71048291", 640.00, "DR"),
        ("08/10/2026", "POS ECOM FLIPKART INTERNET PVT LTD SHOPPING", "POS5519201", 3299.00, "DR"),
        ("08/10/2026", "POS ECOM FLIPKART INTERNET PVT LTD SHOPPING", "POS5519201", 3299.00, "DR"),  # Duplicate edge case
        ("09/10/2026", "ECOM REFUND FLIPKART INTERNET REVERSAL FOR DUPLICATE", "REV5519201", 3299.00, "CR"),
        ("11/10/2026", "UPI/DR/71092837/BHARTI AIRTEL MOBILE & BROADBAND BILL", "UPI71092837", 1199.00, "DR"),
        ("14/10/2026", "ACH-DR-LIC OF INDIA LIFE INSURANCE POLICY PREMIUM", "ACH4410293", 12400.00, "DR"),
        ("17/10/2026", "POS DEBIT MAKEMYTRIP INDIA FLIGHT BOOKING", "POS5529104", 6850.00, "DR"),
        ("20/10/2026", "UPI/DR/71182930/TATA 1MG HEALTHCARE DIAGNOSTIC LAB", "UPI71182930", 1499.00, "DR"),
        ("22/10/2026", "UPI/DR/71201928/BOOKMYSHOW MOVIE TICKETS PVR INOX", "UPI71201928", 820.00, "DR"),
        ("25/10/2026", "IMPS-CR-RAZORPAY SETTLEMENT CLIENT CONSULTING INVOICE", "IMPS9918273", 15000.00, "CR"),
        ("28/10/2026", "NEFT-DR-SBI CARDS AND PAYMENT SERVICES CC BILL", "NEFT8829104", 13600.00, "DR"),
        ("30/10/2026", "DEBIT CARD ANNUAL MAINTENANCE FEE AMC CHARGES", "AMC202610", 590.00, "DR"),
    ]

    table_data = [["Date", "Transaction Remarks", "Ref No", "Amount", "Type", "Balance"]]
    bal = 82000.00
    for dt, desc, ref, amt, t_type in raw_txns:
        bal = round(bal + amt if t_type == "CR" else bal - amt, 2)
        table_data.append(
            [
                dt,
                desc,
                ref,
                f"{amt:,.2f} {t_type}",
                t_type,
                f"{bal:,.2f}",
            ]
        )

    col_widths = [22 * mm, 76 * mm, 26 * mm, 24 * mm, 14 * mm, 24 * mm]
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#8B1E0F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E0")),
                ("ALIGN", (3, 1), (5, -1), "RIGHT"),
                ("ALIGN", (4, 1), (4, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FFF8F6")]),
            ]
        )
    )
    story.append(t)
    doc.build(story)
    return output_path


def generate_sbi_scanned_image_pdf(output_path: str) -> str:
    """Generates a true scanned/image-based PDF (no text layer, slight 0.8° scanner skew, 300 DPI raster) for OCR testing."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Create high-resolution 300 DPI A4 canvas (2480 x 3508 pixels)
    width, height = 2480, 3508
    img = Image.new("RGB", (width, height), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    # Load clean fonts across Windows and Linux at 300 DPI sizes
    font_paths = [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    reg_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    bold_font_path = next((p for p in font_paths if os.path.exists(p)), None)
    reg_font_path = next((p for p in reg_paths if os.path.exists(p)), None)

    f_title = ImageFont.truetype(bold_font_path, 46) if bold_font_path else ImageFont.load_default()
    f_meta = ImageFont.truetype(reg_font_path, 31) if reg_font_path else ImageFont.load_default()
    f_head = ImageFont.truetype(bold_font_path, 30) if bold_font_path else ImageFont.load_default()
    f_row = ImageFont.truetype(reg_font_path, 29) if reg_font_path else ImageFont.load_default()

    y = 140
    draw.text((150, y), "STATE BANK OF INDIA - ACCOUNT STATEMENT", fill=(15, 45, 90), font=f_title)
    y += 85

    meta_texts = [
        "Account Holder Name : PRIYA NAIR",
        "Account Number : 30984716254",
        "IFSC Code : SBIN0004120",
        "Branch Name : Indiranagar Bengaluru",
        "Account Type : Regular Savings Bank Account",
        "Statement Period : 01/10/2026 to 31/10/2026",
        "Opening Balance : 64,500.00",
        "Closing Balance : 88,690.00",
    ]
    for mt in meta_texts:
        draw.text((150, y), mt, fill=(25, 25, 25), font=f_meta)
        y += 52

    y += 45
    draw.line([(140, y), (2340, y)], fill=(40, 40, 40), width=3)
    y += 20
    draw.text((150, y), "Date             Description                                                     Amount            Balance", fill=(15, 45, 90), font=f_head)
    y += 55
    draw.line([(140, y), (2340, y)], fill=(40, 40, 40), width=3)
    y += 30

    scanned_txns = [
        ("01/10/2026", "NEFT CR SBIN0004120 WIPRO LTD SALARY CREDIT", "72,000.00 Cr", "1,36,500.00"),
        ("03/10/2026", "IMPS DR MONTHLY HOUSE RENT APARTMENT LEASE", "21,000.00 Dr", "1,15,500.00"),
        ("05/10/2026", "UPI DR BIGBASKET SUPERMARKET GROCERY SUPPLIES", "2,450.00 Dr", "1,13,050.00"),
        ("08/10/2026", "BBPS DR BESCOM ELECTRICITY BILL PAYMENT", "1,890.00 Dr", "1,11,160.00"),
        ("10/10/2026", "UPI DR GROWW MUTUAL FUND SIP INVESTMENT", "10,000.00 Dr", "1,01,160.00"),
        ("12/10/2026", "POS DEBIT STARBUCKS COFFEE DINING", "480.00 Dr", "1,00,680.00"),
        ("15/10/2026", "UPI DR IRCTC RAIL TICKET BOOKING", "1,540.00 Dr", "99,140.00"),
        ("18/10/2026", "POS DEBIT MYNTRA DESIGNS FASHION SHOPPING", "2,850.00 Dr", "96,290.00"),
        ("21/10/2026", "ATM WDL SBI ATM CASH WITHDRAWAL INDIRANAGAR", "5,000.00 Dr", "91,290.00"),
        ("24/10/2026", "POS DEBIT MANIPAL HOSPITAL OPD CONSULTATION", "1,650.00 Dr", "89,640.00"),
        ("27/10/2026", "UPI AUTOPAY SPOTIFY INDIA MUSIC STREAMING", "119.00 Dr", "89,521.00"),
        ("30/10/2026", "UPI DR ACT FIBERNET BROADBAND INTERNET BILL", "831.00 Dr", "88,690.00"),
    ]

    for dt, desc, amt_str, bal_str in scanned_txns:
        draw.text((150, y), dt, fill=(20, 20, 20), font=f_row)
        draw.text((430, y), desc, fill=(20, 20, 20), font=f_row)
        draw.text((1680, y), amt_str, fill=(20, 20, 20), font=f_row)
        draw.text((2040, y), bal_str, fill=(20, 20, 20), font=f_row)
        y += 72

    # Apply slight realistic scanner skew (0.75 degrees) & mild sensor noise to prove deskew + OCR
    arr = np.array(img)
    center = (width // 2, height // 2)
    rot_mat = cv2.getRotationMatrix2D(center, 0.75, 1.0)
    skewed = cv2.warpAffine(arr, rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderValue=(252, 252, 250))

    pil_skewed = Image.fromarray(skewed)
    # Save directly as an image-only PDF (zero embedded fonts/text streams)
    pil_skewed.save(output_path, "PDF", resolution=300.0)
    return output_path


def generate_all_sample_pdfs(output_dir: str) -> Dict[str, str]:
    """Generates all three benchmark PDF bank statements and returns their paths."""
    os.makedirs(output_dir, exist_ok=True)
    p1 = generate_hdfc_text_pdf(os.path.join(output_dir, "HDFC_Bank_Text_Statement_Oct2026.pdf"))
    p2 = generate_icici_text_pdf(os.path.join(output_dir, "ICICI_Bank_Text_Statement_Oct2026.pdf"))
    p3 = generate_sbi_scanned_image_pdf(os.path.join(output_dir, "SBI_Scanned_Image_Statement_Oct2026.pdf"))
    return {
        "hdfc_text": p1,
        "icici_text": p2,
        "sbi_scanned": p3,
    }
