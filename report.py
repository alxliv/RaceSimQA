"""
PDF Report generation for RaceSim Analyzer.

Generates comprehensive PDF reports combining:
- Executive summary
- Summary metrics comparison table
- Telemetry threshold analysis
- Embedded visualization plots
- AI insights (if available)
"""

import io
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image, PageBreak, KeepTogether, HRFlowable
    )
    from reportlab.graphics.shapes import Drawing, Line
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from analysis import AnalysisResult, Analyzer


# Color scheme
COLORS = {
    "primary": colors.HexColor("#1e40af"),      # Dark blue
    "secondary": colors.HexColor("#3b82f6"),    # Blue
    "success": colors.HexColor("#16a34a"),      # Green
    "warning": colors.HexColor("#f59e0b"),      # Amber
    "danger": colors.HexColor("#dc2626"),       # Red
    "muted": colors.HexColor("#6b7280"),        # Gray
    "light": colors.HexColor("#f3f4f6"),        # Light gray
    "header_bg": colors.HexColor("#1e3a5f"),    # Dark blue
    "row_alt": colors.HexColor("#f8fafc"),      # Very light blue
}

STATUS_COLORS = {
    "excellent": COLORS["success"],
    "good": colors.HexColor("#22c55e"),
    "acceptable": COLORS["warning"],
    "poor": colors.HexColor("#f97316"),
    "fail": COLORS["danger"],
}


def check_reportlab():
    """Check if reportlab is available."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is required for PDF generation. "
            "Install with: pip install reportlab"
        )


@dataclass
class ReportConfig:
    """Configuration for PDF report generation."""
    title: str = "RaceSim Analysis Report"
    subtitle: str = ""
    author: str = "RaceSim Analyzer"
    page_size: tuple = A4
    margin: float = 0.75 * inch
    include_plots: bool = True
    include_ai_analysis: bool = True
    logo_path: Optional[str] = None


class PDFReportGenerator:
    """Generates PDF reports for simulation analysis."""

    def __init__(self, config: ReportConfig = None):
        check_reportlab()
        self.config = config or ReportConfig()
        self.styles = self._create_styles()

    def _create_styles(self) -> dict:
        """Create custom paragraph styles."""
        base_styles = getSampleStyleSheet()

        styles = {
            "title": ParagraphStyle(
                "CustomTitle",
                parent=base_styles["Title"],
                fontSize=24,
                textColor=COLORS["primary"],
                spaceAfter=6,
                alignment=TA_CENTER,
            ),
            "subtitle": ParagraphStyle(
                "CustomSubtitle",
                parent=base_styles["Normal"],
                fontSize=14,
                textColor=COLORS["muted"],
                spaceAfter=20,
                alignment=TA_CENTER,
            ),
            "heading1": ParagraphStyle(
                "CustomH1",
                parent=base_styles["Heading1"],
                fontSize=16,
                textColor=COLORS["primary"],
                spaceBefore=20,
                spaceAfter=10,
                borderWidth=0,
                borderPadding=0,
            ),
            "heading2": ParagraphStyle(
                "CustomH2",
                parent=base_styles["Heading2"],
                fontSize=13,
                textColor=COLORS["secondary"],
                spaceBefore=15,
                spaceAfter=8,
            ),
            "heading3": ParagraphStyle(
                "CustomH3",
                parent=base_styles["Heading3"],
                fontSize=11,
                textColor=COLORS["muted"],
                spaceBefore=10,
                spaceAfter=6,
            ),
            "body": ParagraphStyle(
                "CustomBody",
                parent=base_styles["Normal"],
                fontSize=10,
                leading=14,
                alignment=TA_JUSTIFY,
            ),
            "body_small": ParagraphStyle(
                "CustomBodySmall",
                parent=base_styles["Normal"],
                fontSize=9,
                leading=12,
            ),
            "metric_good": ParagraphStyle(
                "MetricGood",
                parent=base_styles["Normal"],
                fontSize=10,
                textColor=COLORS["success"],
            ),
            "metric_bad": ParagraphStyle(
                "MetricBad",
                parent=base_styles["Normal"],
                fontSize=10,
                textColor=COLORS["danger"],
            ),
            "score_large": ParagraphStyle(
                "ScoreLarge",
                parent=base_styles["Normal"],
                fontSize=36,
                alignment=TA_CENTER,
                spaceBefore=10,
                spaceAfter=10,
            ),
            "status": ParagraphStyle(
                "Status",
                parent=base_styles["Normal"],
                fontSize=14,
                alignment=TA_CENTER,
                spaceAfter=20,
            ),
            "caption": ParagraphStyle(
                "Caption",
                parent=base_styles["Normal"],
                fontSize=9,
                textColor=COLORS["muted"],
                alignment=TA_CENTER,
                spaceBefore=5,
                spaceAfter=15,
            ),
            "code": ParagraphStyle(
                "Code",
                parent=base_styles["Code"],
                fontSize=8,
                leading=10,
                leftIndent=10,
                backColor=COLORS["light"],
            ),
        }

        return styles

    def _create_header(self, canvas, doc):
        """Draw page header."""
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLORS["muted"])
        canvas.drawString(
            self.config.margin,
            self.config.page_size[1] - 0.5 * inch,
            self.config.title
        )
        canvas.drawRightString(
            self.config.page_size[0] - self.config.margin,
            self.config.page_size[1] - 0.5 * inch,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        )
        canvas.restoreState()

    def _create_footer(self, canvas, doc):
        """Draw page footer with page number."""
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLORS["muted"])
        canvas.drawCentredString(
            self.config.page_size[0] / 2,
            0.5 * inch,
            f"Page {doc.page}"
        )
        canvas.restoreState()

    def _header_footer(self, canvas, doc):
        """Combined header and footer callback."""
        self._create_header(canvas, doc)
        self._create_footer(canvas, doc)

    def _create_title_section(self, batch_id: str) -> list:
        """Create title section elements."""
        elements = []

        elements.append(Paragraph(self.config.title, self.styles["title"]))

        subtitle = self.config.subtitle or f"Batch: {batch_id}"
        elements.append(Paragraph(subtitle, self.styles["subtitle"]))

        # Horizontal line
        elements.append(HRFlowable(
            width="100%",
            thickness=1,
            color=COLORS["light"],
            spaceAfter=20
        ))

        return elements

    def _create_executive_summary(
        self,
        result: AnalysisResult,
        telemetry_data: dict = None
    ) -> list:
        """Create executive summary section."""
        elements = []

        elements.append(Paragraph("Executive Summary", self.styles["heading1"]))
        elements.append(Spacer(1, 10))

        # Score and status in a table for better alignment
        score_color = STATUS_COLORS.get(result.status, COLORS["muted"])

        # Create score display as a centered table
        score_table_data = [
            [Paragraph(
                f"<font size='48' color='{score_color.hexval()}'><b>{result.overall_score:.1%}</b></font>",
                ParagraphStyle("ScoreInline", alignment=TA_CENTER)
            )],
            [Paragraph(
                f"<font size='14' color='{score_color.hexval()}'><b>{result.status.upper()}</b></font>",
                ParagraphStyle("StatusInline", alignment=TA_CENTER, spaceBefore=5)
            )],
        ]

        score_table = Table(score_table_data, colWidths=[3*inch])
        score_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))

        # Center the score table
        outer_table = Table([[score_table]], colWidths=[self.config.page_size[0] - 2*self.config.margin])
        outer_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(outer_table)
        elements.append(Spacer(1, 20))

        # Summary stats table
        summary_data = [
            ["Metric", "Value"],
            ["Candidate Batch", result.candidate_batch_id],
            ["Baseline Runs", str(len(result.baseline_run_ids))],
            ["Candidate Runs", str(len(result.candidate_run_ids))],
            ["Requirement Violations", str(len(result.requirement_violations))],
        ]

        if telemetry_data:
            new_violations = len(telemetry_data.get("new_violations", []))
            resolved = len(telemetry_data.get("resolved_violations", []))
            summary_data.append(["New Telemetry Violations", str(new_violations)])
            summary_data.append(["Resolved Telemetry Issues", str(resolved)])

        summary_table = Table(summary_data, colWidths=[2.5*inch, 3*inch])
        summary_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["header_bg"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("PADDING", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["row_alt"]]),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Violations warning box
        if result.requirement_violations:
            elements.append(Paragraph("⚠️ Requirement Violations", self.styles["heading2"]))
            for violation in result.requirement_violations:
                elements.append(Paragraph(
                    f"• {violation}",
                    self.styles["metric_bad"]
                ))
            elements.append(Spacer(1, 10))

        return elements

    def _create_metrics_table(self, result: AnalysisResult) -> list:
        """Create detailed metrics comparison table."""
        elements = []

        elements.append(Paragraph("Metrics Comparison", self.styles["heading1"]))

        # Table header
        table_data = [[
            "Metric",
            "Baseline\n(mean ± std)",
            "Candidate\n(mean ± std)",
            "Delta",
            "Target",
            "Score"
        ]]

        # Table rows
        for name, comp in sorted(result.metric_comparisons.items()):
            unit = comp.requirement.unit if comp.requirement else ""

            baseline_str = f"{comp.baseline_stats.mean:.2f} ± {comp.baseline_stats.std:.2f}"
            candidate_str = f"{comp.candidate_stats.mean:.2f} ± {comp.candidate_stats.std:.2f}"

            delta_sign = "+" if comp.delta_mean >= 0 else ""
            delta_str = f"{delta_sign}{comp.delta_mean:.2f} ({delta_sign}{comp.delta_percent:.1f}%)"

            target_str = f"{comp.requirement.target}" if comp.requirement else "-"
            score_str = f"{comp.score:.0%}"

            indicator = "✓" if comp.improved else "✗"

            table_data.append([
                f"{name} ({unit})" if unit else name,
                baseline_str,
                candidate_str,
                f"{delta_str} {indicator}",
                target_str,
                score_str
            ])

        # Create table
        col_widths = [1.5*inch, 1.2*inch, 1.2*inch, 1.1*inch, 0.7*inch, 0.6*inch]
        table = Table(table_data, colWidths=col_widths)

        # Style the table
        style_commands = [
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["header_bg"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["row_alt"]]),
        ]

        # Color code the score column based on value
        for i, (name, comp) in enumerate(sorted(result.metric_comparisons.items()), start=1):
            if comp.score >= 0.75:
                style_commands.append(("TEXTCOLOR", (-1, i), (-1, i), COLORS["success"]))
            elif comp.score >= 0.5:
                style_commands.append(("TEXTCOLOR", (-1, i), (-1, i), COLORS["warning"]))
            else:
                style_commands.append(("TEXTCOLOR", (-1, i), (-1, i), COLORS["danger"]))

            # Color the delta indicator
            if comp.improved:
                style_commands.append(("TEXTCOLOR", (3, i), (3, i), COLORS["success"]))
            else:
                style_commands.append(("TEXTCOLOR", (3, i), (3, i), COLORS["danger"]))

        table.setStyle(TableStyle(style_commands))
        elements.append(table)
        elements.append(Spacer(1, 20))

        return elements

    def _create_telemetry_section(self, telemetry_data: dict) -> list:
        """Create telemetry analysis section."""
        elements = []

        elements.append(Paragraph("Telemetry Analysis", self.styles["heading1"]))

        # Overview
        elements.append(Paragraph(
            f"Analyzed {telemetry_data['baseline_run_count']} baseline runs and "
            f"{telemetry_data['candidate_run_count']} candidate runs.",
            self.styles["body"]
        ))
        elements.append(Spacer(1, 10))

        # New violations
        new_violations = telemetry_data.get("new_violations", [])
        resolved = telemetry_data.get("resolved_violations", [])

        if new_violations:
            elements.append(Paragraph("⚠️ New Threshold Violations", self.styles["heading2"]))
            for v in new_violations:
                elements.append(Paragraph(f"• {v}", self.styles["metric_bad"]))
            elements.append(Spacer(1, 10))

        if resolved:
            elements.append(Paragraph("✓ Resolved Issues", self.styles["heading2"]))
            for v in resolved:
                elements.append(Paragraph(f"• {v}", self.styles["metric_good"]))
            elements.append(Spacer(1, 10))

        # Threshold crossings table
        candidate_crossings = telemetry_data.get("candidate_threshold_crossings", [])
        if candidate_crossings:
            elements.append(Paragraph("Candidate Threshold Crossings", self.styles["heading2"]))

            table_data = [["Channel", "Threshold", "Severity", "Count", "Avg Duration (m)", "Worst Peak"]]

            for crossing in sorted(candidate_crossings, key=lambda x: (
                {"critical": 0, "warning": 1, "info": 2}.get(x["severity"], 3),
                -x["count"]
            )):
                table_data.append([
                    crossing["channel"],
                    crossing["threshold"],
                    crossing["severity"].upper(),
                    str(crossing["count"]),
                    f"{crossing['avg_duration_m']:.1f}",
                    f"{crossing['worst_peak']:.2f}"
                ])

            table = Table(table_data, colWidths=[1.3*inch, 1.3*inch, 0.8*inch, 0.6*inch, 1*inch, 0.9*inch])

            style_commands = [
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["header_bg"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (2, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["row_alt"]]),
            ]

            # Color code severity
            for i, crossing in enumerate(sorted(candidate_crossings, key=lambda x: (
                {"critical": 0, "warning": 1, "info": 2}.get(x["severity"], 3),
                -x["count"]
            )), start=1):
                if crossing["severity"] == "critical":
                    style_commands.append(("TEXTCOLOR", (2, i), (2, i), COLORS["danger"]))
                elif crossing["severity"] == "warning":
                    style_commands.append(("TEXTCOLOR", (2, i), (2, i), COLORS["warning"]))

            table.setStyle(TableStyle(style_commands))
            elements.append(table)
            elements.append(Spacer(1, 15))

        # Channel differences summary
        channels = telemetry_data.get("channels", {})
        if channels:
            elements.append(Paragraph("Channel Differences Summary", self.styles["heading2"]))

            table_data = [["Channel", "Baseline Range", "Candidate Range", "RMS Delta", "Max Delta Position"]]

            for name, data in sorted(channels.items()):
                bl_range = data.get("baseline_mean_range", [0, 0])
                cd_range = data.get("candidate_mean_range", [0, 0])
                table_data.append([
                    name,
                    f"{bl_range[0]:.2f} - {bl_range[1]:.2f}",
                    f"{cd_range[0]:.2f} - {cd_range[1]:.2f}",
                    f"{data.get('rms_delta', 0):.4f}",
                    f"{data.get('max_delta_position_m', 0):.0f} m"
                ])

            table = Table(table_data, colWidths=[1.3*inch, 1.2*inch, 1.2*inch, 0.9*inch, 1.1*inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["header_bg"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["row_alt"]]),
            ]))
            elements.append(table)

        return elements

    def _create_plots_section(self, plot_paths: list[str]) -> list:
        """Create section with embedded plots."""
        elements = []

        elements.append(PageBreak())
        elements.append(Paragraph("Visualization", self.styles["heading1"]))

        # Calculate available width
        available_width = self.config.page_size[0] - 2 * self.config.margin

        for plot_path in plot_paths:
            if not os.path.exists(plot_path):
                continue

            # Get plot filename for caption
            filename = os.path.basename(plot_path)
            caption = filename.replace("_", " ").replace(".png", "").title()

            # Determine appropriate size based on plot type
            if "dashboard" in filename.lower():
                # Dashboard gets full width
                img_width = available_width
                img_height = available_width * 0.75  # Approximate aspect ratio
            elif "violations" in filename.lower():
                img_width = available_width
                img_height = available_width * 0.9
            else:
                # Other plots
                img_width = available_width * 0.9
                img_height = img_width * 0.6

            try:
                img = Image(plot_path, width=img_width, height=img_height)
                img.hAlign = "CENTER"

                elements.append(KeepTogether([
                    Paragraph(caption, self.styles["heading2"]),
                    img,
                    Paragraph(f"Figure: {caption}", self.styles["caption"]),
                ]))
                elements.append(Spacer(1, 10))

            except Exception as e:
                elements.append(Paragraph(
                    f"Error loading plot: {plot_path} - {str(e)}",
                    self.styles["body"]
                ))

        return elements

    def _create_ai_section(self, ai_analysis: str) -> list:
        """Create AI analysis section with improved styling and table support."""
        elements = []

        elements.append(PageBreak())
        elements.append(Paragraph("AI Analysis", self.styles["heading1"]))
        elements.append(Spacer(1, 10))

        if not ai_analysis:
            return elements

        # Pre-process: standardized line endings
        lines = ai_analysis.replace("\r\n", "\n").split("\n")

        buffer = []
        in_table = False
        table_lines = []

        def flush_buffer():
            if not buffer:
                return
            # Join lines in the buffer to form a paragraph
            text = " ".join(buffer).strip()
            if text:
                self._add_markdown_paragraph(text, elements)
            buffer.clear()

        def flush_table():
            if table_lines:
                try:
                    table_flowable = self._parse_markdown_table(table_lines)
                    if table_flowable:
                        elements.append(table_flowable)
                        elements.append(Spacer(1, 12))
                except Exception:
                    # Fallback if table parsing fails
                    clean_lines = [l.strip() for l in table_lines]
                    self._add_markdown_paragraph("\n".join(clean_lines), elements)
                table_lines.clear()

        for line in lines:
            line = line.strip()

            # Empty lines trigger flush (paragraph break)
            if not line:
                flush_buffer()
                if in_table:
                    flush_table()
                    in_table = False
                continue

            # Table detection
            if line.startswith("|"):
                if not in_table:
                    flush_buffer()
                    in_table = True
                table_lines.append(line)
                continue

            # End of table
            if in_table:
                flush_table()
                in_table = False

            # Header detection (starts new block)
            if line.startswith("#"):
                flush_buffer()
                self._add_markdown_paragraph(line, elements)
                continue

            # List detection (starts new block)
            if line.startswith("- ") or line.startswith("* "):
                flush_buffer()
                self._add_markdown_list_item(line, elements)
                continue

            # Separator line detection
            if line.startswith("---") or line.startswith("***"):
                flush_buffer()
                elements.append(HRFlowable(width="100%", thickness=1, color=COLORS["light"], spaceAfter=12))
                continue

            # Normal text accumulation
            buffer.append(line)

        # Final flush
        flush_buffer()
        if in_table:
            flush_table()

        return elements

    def _escape_xml(self, text: str) -> str:
        """Escape XML/HTML special characters."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _convert_markdown_bold(self, text: str) -> str:
        """
        Convert **bold** to <b>bold</b> HTML.
        Expects text to be ALREADY escaped for XML chars,
        or assumes input does not contain tags that shouldn't be there.
        """
        # We will count on _process_text doing the escaping first
        result = []
        parts = text.split("**")

        for i, part in enumerate(parts):
            if i % 2 == 1: # Inside **
                result.append(f"<b>{part}</b>")
            else:
                result.append(part)

        return "".join(result)

    def _process_text(self, text: str) -> str:
        """Escape XML and apply markdown formatting."""
        safe_text = self._escape_xml(text)
        return self._convert_markdown_bold(safe_text)

    def _add_markdown_paragraph(self, text: str, elements: list):
        """Process and add a markdown paragraph/header."""
        processed_text = self._process_text(text)

        if text.startswith("# "):
            elements.append(Paragraph(processed_text[2:], self.styles["heading1"]))
        elif text.startswith("## "):
            elements.append(Paragraph(processed_text[3:], self.styles["heading2"]))
        elif text.startswith("### "):
            elements.append(Paragraph(processed_text[4:], self.styles["heading3"]))
        else:
            elements.append(Paragraph(processed_text, self.styles["body"]))
            elements.append(Spacer(1, 6))

    def _add_markdown_list_item(self, text: str, elements: list):
        """Process and add a markdown list item."""
        # Strip the bullet marker
        if text.startswith("- "):
            content = text[2:]
        elif text.startswith("* "):
            content = text[2:]
        else:
            content = text

        processed_content = self._process_text(content)
        elements.append(Paragraph(f"• {processed_content}", self.styles["body"], bulletText="•"))
        elements.append(Spacer(1, 3))

    def _parse_markdown_table(self, lines: list[str]):
        """Parse markdown table text into a ReportLab Table."""
        data = []

        for i, line in enumerate(lines):
            # Strip outer pipes if present
            content = line.strip()
            if content.startswith("|"): content = content[1:]
            if content.endswith("|"): content = content[:-1]

            cells = [c.strip() for c in content.split("|")]

            # Check for separator line (e.g. ---|:---|---)
            is_separator = all(set(c) <= set("-:| ") for c in cells) and len(cells) > 0
            if is_separator:
                continue

            # Process cells
            row_data = []
            for cell in cells:
                # Process formatting inside cell
                formatted_cell = self._process_text(cell)
                row_data.append(Paragraph(formatted_cell, self.styles["body_small"]))

            data.append(row_data)

        if not data:
            return None

        # Create table with automatic (but finite) width
        # Simple approach: equal column widths filling the page
        available_width = self.config.page_size[0] - 2 * self.config.margin
        num_cols = max(len(row) for row in data)
        if num_cols == 0: return None

        col_width = available_width / num_cols

        table = Table(data, colWidths=[col_width] * num_cols)

        style = TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, COLORS["light"]),
            ('BACKGROUND', (0, 0), (-1, 0), COLORS["header_bg"]),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, COLORS["row_alt"]]),
        ])
        table.setStyle(style)

        return table

    def generate(
        self,
        output_path: str,
        analysis_result: AnalysisResult,
        telemetry_data: dict = None,
        plot_paths: list[str] = None,
        ai_analysis: str = None,
    ) -> str:
        """
        Generate complete PDF report.

        Args:
            output_path: Path for output PDF file
            analysis_result: AnalysisResult from Analyzer
            telemetry_data: Optional dict from TelemetryAnalyzer.to_dict()
            plot_paths: Optional list of plot image paths to embed
            ai_analysis: Optional AI analysis text

        Returns:
            Path to generated PDF
        """
        doc = SimpleDocTemplate(
            output_path,
            pagesize=self.config.page_size,
            leftMargin=self.config.margin,
            rightMargin=self.config.margin,
            topMargin=self.config.margin + 0.3*inch,
            bottomMargin=self.config.margin + 0.3*inch,
            title=self.config.title,
            author=self.config.author,
        )

        elements = []

        # Title section
        elements.extend(self._create_title_section(analysis_result.candidate_batch_id))

        # Executive summary
        elements.extend(self._create_executive_summary(analysis_result, telemetry_data))

        # Metrics table
        elements.extend(self._create_metrics_table(analysis_result))

        # Telemetry section
        if telemetry_data:
            elements.append(PageBreak())
            elements.extend(self._create_telemetry_section(telemetry_data))

        # Plots section
        if plot_paths and self.config.include_plots:
            elements.extend(self._create_plots_section(plot_paths))

        # AI analysis section
        if ai_analysis and self.config.include_ai_analysis:
            elements.extend(self._create_ai_section(ai_analysis))

        # Build PDF
        doc.build(elements, onFirstPage=self._header_footer, onLaterPages=self._header_footer)

        return output_path


def generate_report(
    output_path: str,
    analysis_result: AnalysisResult,
    telemetry_data: dict = None,
    plot_paths: list[str] = None,
    ai_analysis: str = None,
    title: str = None,
) -> str:
    """
    Convenience function to generate a PDF report.

    Args:
        output_path: Path for output PDF
        analysis_result: AnalysisResult object
        telemetry_data: Optional telemetry comparison data
        plot_paths: Optional list of plot image paths
        ai_analysis: Optional AI analysis text
        title: Optional custom title

    Returns:
        Path to generated PDF
    """
    config = ReportConfig(
        title=title or "RaceSim Analysis Report",
        subtitle=f"Batch: {analysis_result.candidate_batch_id}",
    )

    generator = PDFReportGenerator(config)
    return generator.generate(
        output_path,
        analysis_result,
        telemetry_data,
        plot_paths,
        ai_analysis,
    )