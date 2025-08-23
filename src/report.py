import os
from fpdf import FPDF

def generate_pdf_report(roc_pr_path, feature_paths, cm_paths, output_path="outputs/reports/model_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Model Evaluation Report", ln=True, align='C')

    # ROC & PR
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "ROC & Precision-Recall Curves", ln=True)
    pdf.image(roc_pr_path, x=10, y=25, w=180)

    # Feature importance
    for path in feature_paths:
        pdf.add_page()
        title = os.path.basename(path).replace('_',' ').replace('.png','').title()
        pdf.cell(0, 10, title, ln=True)
        pdf.image(path, x=10, y=25, w=180)

    # Confusion matrix
    for path in cm_paths:
        pdf.add_page()
        title = os.path.basename(path).replace('_',' ').replace('.png','').title()
        pdf.cell(0, 10, title, ln=True)
        pdf.image(path, x=10, y=25, w=180)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    print(f"✅ Saved PDF report: {output_path}")
