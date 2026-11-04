# Assessment Document
## Bank Statement Processing & Classification System

## 1. Objective

The objective of this assessment is to design and develop a fully functional prototype system capable of
processing bank statements in PDF format (both image-based and text-based), extracting relevant financial data,
classifying transactions using non-LLM approaches, and exporting the processed data into structured formats
such as Excel or CSV.

## 2. Problem Statement

The system must accept bank statements in PDF format, which may be either text-based or image-based
(scanned documents). It should accurately extract account details and transaction data, including date,
description, debit/credit amount, and balance.

After extraction, the system must perform transaction classification based on the transaction description. The
classification must not rely on Large Language Models (LLMs) but instead use heuristic methods or traditional
machine learning approaches.

Finally, the system should allow users to export processed and classified data into Excel or CSV format for further
analysis.

The solution should be designed by considering multiple real-world scenarios, edge cases, and usability features
to ensure it evolves into a robust and production-ready prototype.

## 3. Scope of Work

### 3.1 Input Handling

- Accept PDF bank statements from multiple banks
- Support:
  - Text-based PDFs
  - Image-based PDFs
- Handle multiple formats and layouts across banks

### 3.2 Data Extraction

Extract:

- Account holder details (name, account number, IFSC, etc.)
- Transaction table data:
  - Date
  - Description
  - Debit amount
  - Credit amount
  - Balance

### 3.2 Classification Engine (Non-LLM)

Don’t use LLM for classification, The system must classify transactions using:

- Heuristic Rule-Based Approach
- Traditional Machine Learning Models

## 4. Output Requirements

Export processed data in:

- Excel (.xlsx)
- CSV (.csv)

Output should include:

- Extracted transaction details
- Assigned category

## 5. Key Features

- Upload and process bank statements
- Auto-detect PDF type (text vs image)
- Accurate data extraction
- Transaction classification without LLM
- Export to Excel/CSV

Note: Consider advanced features, real-world scenarios, and edge cases while designing the solution, and
ensure they are implemented in the prototype.

## 6. Conclusion

This assessment requires building a practical, scalable, and intelligent system for processing bank statements.
The focus is on designing a robust extraction pipeline and a reliable classification engine without relying on
LLMs. The solution should demonstrate strong handling of real-world variations, extensibility for future
improvements, and readiness for production-level enhancements.

