"""
app.py
======
Supply Chain Analytics Inventory Decision Agent Dashboard & Application.

Provides:
1. Streamlit Web UI:
   - Natural Language Query Interface with 1-click sample prompts for all 5 models.
   - Visual 7-Stage Architectural Trace:
     Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
   - Executive Decision Cards, Mathematical KaTeX Derivations, and Altair Cost Visualizations.
   - Direct Parameter Calculator tabs for what-if sensitivity analysis.
2. CLI Interactive Mode:
   - Command-line interactive agent interface when run directly via `python app.py`.
"""

import sys
import re
from typing import Optional
import pandas as pd
import numpy as np

from inventory_agent import (
    InventoryDecisionAgent,
    InventoryModelType,
    AgentResponse
)
from inventory_tools import (
    calculate_eoq,
    calculate_quantity_discount,
    calculate_rop,
    calculate_periodic_review,
    calculate_safety_stock_rop,
    calculate_fixed_period_review,
    calculate_newsvendor,
    generate_eoq_cost_curve,
    generate_newsvendor_curve,
    validate_inventory_csv_columns,
    analyze_inventory_dataframe,
    validate_inventory_classification_columns,
    classify_inventory_multicriteria
)


# =============================================================================
# Sample Questions for Quick Demonstration
# =============================================================================

SAMPLE_QUERIES = {
    "1. Economic Order Quantity (EOQ)": (
        "We have an annual demand of 12,000 units. Ordering cost is $60 per order and "
        "holding cost is $2.50 per unit per year. Unit purchase price is $15. What is our optimal EOQ?"
    ),
    "2. Quantity Discount": (
        "Our demand is 1,000 units per year with ordering cost of $40. Holding rate is 20%. "
        "The supplier offers discounts: 1-499 units at $5.00, 500-999 units at $4.50, and 1000+ units at $4.00. "
        "What quantity should we order to minimize total annual cost?"
    ),
    "3. Safety Stock & Reorder Point": (
        "Calculate safety stock and reorder point for daily demand of 100 units with standard deviation "
        "of 20 units. Lead time is 4 days and desired cycle service level is 95%."
    ),
    "4. Fixed-Period Review": (
        "We operate a periodic review model reviewed every 7 days. Lead time is 3 days. "
        "Average demand is 50 units per day with standard deviation of 10. Desired service level is 95%. "
        "Current inventory on hand is 150 units and on order is 100 units. How many units should we order?"
    ),
    "5. Newsvendor (Single-Period)": (
        "A retailer sells ski jackets for $100 that cost $40 to buy. Unsold jackets salvage for $10. "
        "Demand is normally distributed with mean 1000 and standard deviation 200. "
        "What is the optimal order quantity under the newsvendor model?"
    )
}


# =============================================================================
# CLI Interface Mode
# =============================================================================

def run_cli():
    """Interactive command-line interface for the Inventory Decision Agent."""
    agent = InventoryDecisionAgent()
    print("=" * 80)
    print("  SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT (CLI MODE)")
    print("  Architecture: Question -> Identify -> Extract -> Tool -> Calc -> Verify -> Explain")
    print("=" * 80)
    print("\nTip: To run the interactive web application, run:")
    print("     streamlit run app.py\n")

    print("Quick Sample Queries:")
    for idx, (label, text) in enumerate(SAMPLE_QUERIES.items(), 1):
        print(f"  [{idx}] {label}")

    while True:
        print("\n" + "-" * 80)
        user_input = input("Enter a sample number [1-5], type your own question, or 'exit' to quit:\n> ").strip()

        if not user_input or user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting Inventory Decision Agent. Goodbye!")
            break

        if user_input in ["1", "2", "3", "4", "5"]:
            idx = int(user_input) - 1
            query = list(SAMPLE_QUERIES.values())[idx]
            print(f"\n[Selected Sample Question]:\n{query}\n")
        else:
            query = user_input

        print("\nProcessing through 7-Stage Agent Pipeline...\n")
        response = agent.process_query(query)

        # Print Traces
        print("STAGE EXECUTION TRACE:")
        for trace in response.step_traces:
            print(f"  [Stage {trace.step_number}] {trace.step_name}: {trace.summary}")

        print("\n" + "=" * 50)
        print("EXECUTIVE RECOMMENDATION:")
        print("=" * 50)
        print(response.executive_recommendation.replace("**", ""))

        if response.verification_checks:
            print("\nVERIFICATION CHECKS:")
            for check in response.verification_checks:
                status_icon = "✓" if check.status == "PASS" else "!"
                print(f"  [{status_icon}] {check.name} ({check.status}): {check.message}")

        if response.success:
            print("\n" + "=" * 50)
            print("STUDENT VERIFICATION BREAKDOWN (8 MANDATORY SECTIONS):")
            print("=" * 50)
            print(response.detailed_explanation)

            print("\n" + "=" * 50)
            print("CALCULATION RESULTS (RAW METRICS):")
            print("=" * 50)
            for k, v in response.calculation_result.items():
                if k not in ["model", "tier_evaluations", "inputs_extracted", "variable_definitions", "formula", "formula_with_substituted_values", "intermediate_calculations", "final_answer", "verification_checks", "assumptions"]:
                    print(f"  {k:35}: {v}")


# =============================================================================
# Custom HTML/CSS Card Metric Renderer
# =============================================================================

def render_metric_card(label: str, value: str, subtext: Optional[str] = None) -> str:
    """
    Renders a custom HTML/CSS card for analytical metrics with clean text wrapping,
    avoiding default st.metric ellipsis truncation.
    """
    subtext_html = (
        f"<br><span style='font-size: 0.80rem; font-weight: 600; color: #4b5563;'>{subtext}</span>"
        if subtext else ""
    )
    return (
        f'<div style="background-color: #f8f9fa; border-radius: 10px; padding: 12px 8px; '
        f'text-align: center; border: 1px solid #e9ecef; min-height: 88px; display: flex; '
        f'flex-direction: column; justify-content: center;">'
        f'<div style="font-size: 0.72rem; color: #6c757d; text-transform: uppercase; font-weight: 600; '
        f'letter-spacing: 0.5px; margin-bottom: 4px;">{label}</div>'
        f'<div style="font-size: 1.05rem; font-weight: 700; color: #1f2937; white-space: normal; '
        f'word-wrap: break-word; line-height: 1.35;">'
        f'{value}{subtext_html}'
        f'</div></div>'
    )


# =============================================================================
# Streamlit Web UI Mode
# =============================================================================

def run_streamlit():
    import streamlit as st
    import altair as alt

    st.set_page_config(
        page_title="Supply Chain Analytics | Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize agent in session state
    if "agent" not in st.session_state:
        st.session_state.agent = InventoryDecisionAgent()

    # App Header
    st.title("📦 Supply Chain Analytics Inventory Decision Agent")
    st.markdown(
        "An AI-powered decision support agent implementing the **7-Stage Supply Chain Architecture** "
        "for deterministic inventory optimization."
    )

    # Architectural Pipeline Banner
    pipeline_html = """
    <div style="background-color: #f0f4f9; border-radius: 8px; padding: 12px 18px; margin-bottom: 20px; border-left: 5px solid #1a73e8;">
        <span style="font-weight: 600; color: #1a73e8;">Pipeline: </span>
        <span style="color: #3c4043; font-size: 0.9em;">
            <b>1. Customer Question</b> &nbsp;→&nbsp;
            <b>2. Identify Model</b> &nbsp;→&nbsp;
            <b>3. Extract Inputs</b> &nbsp;→&nbsp;
            <b>4. Select Python Tool</b> &nbsp;→&nbsp;
            <b>5. Calculate</b> &nbsp;→&nbsp;
            <b>6. Verify</b> &nbsp;→&nbsp;
            <b>7. Explain</b>
        </span>
    </div>
    """
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # Sidebar: Model Directory & Guide
    with st.sidebar:
        st.header("📚 Supported Models")
        st.markdown("""
        1. **Economic Order Quantity (EOQ)**
           - Trade-off between annual holding cost and ordering/setup cost.
        2. **Quantity Discount Model**
           - All-units price break evaluation and global cost minimization.
        3. **Safety Stock & Reorder Point (ROP)**
           - Continuous review $(s, Q)$ with normal lead-time demand uncertainty.
        4. **Fixed-Period Review Model**
           - Periodic review $(P, S)$ Order-Up-To system protecting over $P + L$.
        5. **Newsvendor Model**
           - Single-period critical fractile optimization for seasonal/perishable goods.
        """)

        st.divider()
        st.caption("BAN 517 Supply Chain Analytics Project")
        st.caption("Deterministic Numerical Engine: Python `inventory_tools.py`")

        # Section: Data Upload (Last section in the sidebar)
        st.divider()
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        # Sample Datasets for Testing
        sample_classification_csv = (
            "Item Number,Item Name,UoM,Sales ($),POS Hits\n"
            "#216,Industrial Motor 5HP,EA,4694447,186\n"
            "#149,Hydraulic Pump Unit,EA,4551669,55\n"
            "#191,Programmable Logic Controller,EA,4347291,317\n"
            "#425,Servo Drive Controller,EA,4192948,48\n"
            "#136,Digital Sensor Module,EA,3507850,170\n"
            "#411,Pneumatic Cylinder 50mm,EA,1278763,76\n"
            "#435,Control Valve Assembly,EA,1273914,10\n"
            "#483,Pressure Transducer,EA,1251174,7\n"
            "#406,Flow Meter 2-Inch,EA,1216696,227\n"
            "#481,Optical Proximity Switch,EA,1090841,308\n"
            "#1432,Rotary Encoder 1000PPR,EA,1084760,97\n"
            "#490,Thermal Overload Relay,EA,1062341,257\n"
            "#1129,Magnetic Contactor 40A,EA,1043602,86\n"
            "#240,Industrial Bearing 6205,EA,996214,92\n"
            "#488,Limit Switch Heavy Duty,EA,890520,64\n"
            "#244,Stainless Steel Fastener Set,PK,754200,42\n"
            "#135,Terminal Block 10-Pole,EA,621000,81\n"
            "#1164,V-Belt Section B,EA,589400,60\n"
            "#1240,Filter Regulator Lubricator,EA,310500,28\n"
            "#464,Wire Duct 2x2 Gray,FT,185200,15\n"
            "#210,Cable Gland M20,PK,142000,35\n"
            "#107,Replacement Fuse 10A,PK,98500,12\n"
            "#139,Warning Label Decal,PK,65200,8\n"
            "#1191,Hex Nut M8 100pk,PK,42100,6\n"
            "#161,Flat Washer M8 100pk,PK,28400,5\n"
            "#371,O-Ring Nitrile 70A,PK,19500,4\n"
            "#398,Grounding Lug Copper,EA,14200,90\n"
            "#427,Rubber Grommet Assortment,PK,8500,3\n"
            "#463,Zip Ties 8-Inch 100pk,PK,4200,2\n"
        )
        sample_valuation_csv = (
            "Product,Category,Units,Unit Cost\n"
            "Widget Alpha,Electronics,450,15.50\n"
            "Sensor Beta,Electronics,1200,8.25\n"
            "Steel Bolt Pack,Hardware,3500,1.10\n"
            "Torque Wrench,Tools,180,45.00\n"
            "Safety Goggles,PPE,850,5.75\n"
        )

        st.download_button(
            label="📥 Download Classification Sample CSV",
            data=sample_classification_csv,
            file_name="sample_classification_inventory.csv",
            mime="text/csv",
            help="Download a formatted sample CSV with Item Number, Item Name, UoM, Sales ($), and POS Hits to test Multi-Criteria Classification.",
            use_container_width=True
        )
        st.download_button(
            label="📥 Download Valuation Sample CSV",
            data=sample_valuation_csv,
            file_name="sample_valuation_inventory.csv",
            mime="text/csv",
            help="Download a formatted sample CSV with Product, Category, Units, and Unit Cost to test Inventory Valuation.",
            use_container_width=True
        )

    # -------------------------------------------------------------------------
    # Uploaded CSV Data Processing and Analysis
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing_class_cols, class_col_map = validate_inventory_classification_columns(uploaded_df)
            missing_val_cols = validate_inventory_csv_columns(uploaded_df)

            # -----------------------------------------------------------------
            # BRANCH 1: Multi-Criteria Inventory Classification (Sales & POS Hits)
            # -----------------------------------------------------------------
            if not missing_class_cols:
                st.header("🏷️ Multicriteria Inventory Classification (Sales & POS Hits)")
                st.caption(
                    "Deterministic 3-Stage Multi-Criteria Classification using **Sales ($)** and **POS Hits** | "
                    "BAN 517 Supply Chain Analytics Methodology"
                )

                # Methodology Step-by-Step Overview Callout
                st.markdown("""
                <div style="background-color: #f8f9fa; border-left: 5px solid #1a73e8; padding: 14px 18px; border-radius: 6px; margin-bottom: 16px; font-size: 0.93rem; color: #1f2937;">
                    <div style="font-weight: 700; color: #1a73e8; margin-bottom: 6px; font-size: 1.0rem;">📚 Multicriteria Classification Methodology</div>
                    <ul style="margin: 0; padding-left: 20px; line-height: 1.6;">
                        <li><b>Step 1 — Classify Sales and POS Hits Separately</b>: Rank items independently using Pareto thresholds:
                            <b>A</b> (0 to 60% cumulative share; &lt; 61%), <b>B</b> (61 to 80% cumulative share; &lt; 81%),
                            <b>C</b> (81 to 90% cumulative share; &lt; 91%), and <b>D</b> (91 to 100% cumulative share; ≥ 91%).
                            Convert each class to a numerical score: <b>A = 4, B = 3, C = 2, D = 1</b>.</li>
                        <li><b>Step 2 — Calculate Combined Weighted Score</b>: Weighted at <b>65% for Sales</b> and <b>35% for POS Hits</b>:<br>
                            <code>Combined Weighted Score = 0.65(Sales Score) + 0.35(Hits Score)</code><br>
                            <i>Example: If Sales Class is A (4) and POS Hits Class is B (3), Combined Weighted Score = 0.65(4) + 0.35(3) = <b>3.65</b>.</i></li>
                        <li><b>Step 3 — Assign Final Inventory Class</b>:
                            <b>Class A</b>: 4.00 to 3.25 | <b>Class B</b>: 3.24 to 2.50 | <b>Class C</b>: 2.49 to 1.75 | <b>Class D</b>: less than 1.75.</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

                # Weighting Controls (Default: 65% Sales, 35% POS Hits)
                with st.expander("⚙️ Classification Parameters & Sensitivity Weighting Controls (Click to inspect / customize)", expanded=False):
                    w_col1, w_col2, w_col3 = st.columns([2, 2, 3])
                    sales_weight_pct = w_col1.slider(
                        "Sales ($) Criteria Weight (%)",
                        min_value=10, max_value=90, value=65, step=5,
                        help="Standard BAN 517 baseline is 65% for Sales ($)."
                    )
                    hits_weight_pct = 100 - sales_weight_pct
                    w_col2.metric("POS Hits Criteria Weight (%)", f"{hits_weight_pct}%")
                    w_col3.info(
                        f"**Current Scoring Formula**:\n"
                        f"Combined Weighted Score = ({sales_weight_pct/100:.2f} × Sales Score) + ({hits_weight_pct/100:.2f} × Hits Score)\n\n"
                        f"**Pareto Thresholds**:\n"
                        f"• A: 0–60% (< 61%) | B: 61–80% (< 81%)\n• C: 81–90% (< 91%) | D: 91–100% (≥ 91%)\n\n"
                        f"**Final Class Cutoffs**:\n"
                        f"• A: 4.00 to 3.25\n• B: 3.24 to 2.50\n• C: 2.49 to 1.75\n• D: < 1.75"
                    )

                w_sales = sales_weight_pct / 100.0
                w_hits = hits_weight_pct / 100.0

                # Execute deterministic classification analysis
                class_res = classify_inventory_multicriteria(
                    uploaded_df,
                    sales_weight=w_sales,
                    hits_weight=w_hits,
                    sales_cutoffs=(0.61, 0.81, 0.91),
                    hits_cutoffs=(0.61, 0.81, 0.91),
                    score_cutoffs={"A": 3.25, "B": 2.50, "C": 1.75}
                )
                classified_df = class_res["df"]
                class_summary = class_res["class_summary"]

                # Dataset Overview Metrics
                st.subheader("Dataset Dimensions & Summary")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Total Classified Items", f"{class_res['total_skus']:,}")
                d2.metric("Total Catalog Sales", f"${class_res['total_sales']:,.2f}")
                d3.metric("Total POS Hits", f"{class_res['total_hits']:,}")
                d4.metric("Avg Annual Sales / Item", f"${class_res['avg_sales_per_sku']:,.2f}")

                st.write("")

                # -------------------------------------------------------------
                # REQUIRED CLASSIFICATION SUMMARY
                # -------------------------------------------------------------
                st.subheader("📋 Required Classification Summary")
                st.caption(
                    "Reporting the number of items in each final category (A, B, C, and D) "
                    "and summarizing total Sales for all items in each category."
                )

                # 4 KPI Cards for Categories A, B, C, D
                kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
                cards_meta = [
                    (kpi_a, "Final Category A", "#1b5e20", "#e8f5e9", "4.00 to 3.25"),
                    (kpi_b, "Final Category B", "#0d47a1", "#e3f2fd", "3.24 to 2.50"),
                    (kpi_c, "Final Category C", "#e65100", "#fff3e0", "2.49 to 1.75"),
                    (kpi_d, "Final Category D", "#b71c1c", "#ffebee", "< 1.75")
                ]
                for (col, label, border_c, bg_c, score_range), (_, row_data) in zip(cards_meta, class_summary.iterrows()):
                    with col:
                        st.markdown(
                            f"""<div style="background-color: {bg_c}; border-top: 4px solid {border_c}; padding: 14px 16px; border-radius: 8px; margin-bottom: 8px;">
                                <div style="font-size: 0.8rem; font-weight: 700; color: {border_c}; text-transform: uppercase;">{label}</div>
                                <div style="font-size: 0.75rem; color: #6c757d; margin-top: 2px;">Score: {score_range}</div>
                                <div style="font-size: 1.35rem; font-weight: 800; color: #212529; margin-top: 6px;">{int(row_data['Number of Items']):,} Items <span style="font-size: 0.85rem; font-weight: 500; color: #495057;">({row_data['% of Total Items']:.1f}%)</span></div>
                                <div style="font-size: 1.05rem; font-weight: 700; color: #1f2937; margin-top: 6px;">Total Sales: ${row_data['Total Sales ($)']:,.2f} <span style="font-size: 0.8rem; font-weight: normal; color: #6c757d;">({row_data['% of Total Sales']:.1f}%)</span></div>
                                <div style="font-size: 0.85rem; color: #495057; margin-top: 4px;">POS Hits: {int(row_data['Total POS Hits']):,} <span style="font-size: 0.8rem; color: #6c757d;">({row_data['% of POS Hits']:.1f}%)</span></div>
                            </div>""",
                            unsafe_allow_html=True
                        )

                st.write("")

                # Required Summary Data Table
                st.dataframe(
                    class_summary[[
                        "Category (Class)",
                        "Number of Items",
                        "% of Total Items",
                        "Total Sales ($)",
                        "% of Total Sales",
                        "Total POS Hits",
                        "% of POS Hits",
                        "Combined Score Range",
                        "Recommended Inventory Policy"
                    ]],
                    column_config={
                        "Category (Class)": st.column_config.TextColumn("Final Category", width="small"),
                        "Number of Items": st.column_config.NumberColumn("Number of Items", format="%d"),
                        "% of Total Items": st.column_config.NumberColumn("% of Total Items", format="%.1f%%"),
                        "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%,.2f"),
                        "% of Total Sales": st.column_config.NumberColumn("% of Total Sales", format="%.1f%%"),
                        "Total POS Hits": st.column_config.NumberColumn("Total POS Hits", format="%d"),
                        "% of POS Hits": st.column_config.NumberColumn("% of POS Hits", format="%.1f%%"),
                        "Combined Score Range": st.column_config.TextColumn("Score Range", width="small"),
                        "Recommended Inventory Policy": st.column_config.TextColumn("Strategic Inventory Policy", width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )

                st.write("")

                # 2D Cross-Classification Matrix
                st.subheader("2D Cross-Classification Matrix (Sales Class × POS Hits Class)")
                st.caption("Rows represent Sales ($) Pareto Class (0-60% A, 61-80% B, 81-90% C, 91-100% D); Columns represent POS Hits Pareto Class.")

                crosstab_view = st.radio(
                    "Select Cross-Tabulation Metric:",
                    ["Number of Items (SKU Count)", "Total Sales ($)"],
                    horizontal=True
                )
                if "Number of Items" in crosstab_view:
                    st.dataframe(class_res["crosstab_counts"], use_container_width=True)
                else:
                    st.dataframe(
                        class_res["crosstab_sales"].style.format("${:,.2f}"),
                        use_container_width=True
                    )

                st.write("")

                # Complete Classified DataFrame & Export
                st.subheader("Classified Inventory Master DataFrame")
                col_dl1, col_dl2 = st.columns([3, 1])
                with col_dl1:
                    st.caption(
                        f"Showing all {len(classified_df)} items with individual Sales/POS Hits Pareto classes, "
                        f"individual scores, Combined Weighted Score, and Final Inventory Class."
                    )
                with col_dl2:
                    csv_export = classified_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Classified CSV",
                        data=csv_export,
                        file_name="classified_inventory_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                st.dataframe(
                    classified_df[[
                        "Item Number",
                        "Item Name",
                        "UoM",
                        "Sales ($)",
                        "POS Hits",
                        "Sales Cumulative Pct",
                        "Sales Class",
                        "Sales Score",
                        "POS Hits Cumulative Pct",
                        "POS Hits Class",
                        "POS Hits Score",
                        "Combined Weighted Score",
                        "Final Inventory Class"
                    ]],
                    column_config={
                        "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%,.2f"),
                        "POS Hits": st.column_config.NumberColumn("POS Hits", format="%d"),
                        "Sales Cumulative Pct": st.column_config.NumberColumn("Sales Cum %", format="%.0f%%"),
                        "Sales Score": st.column_config.NumberColumn("Sales Score", format="%.0f"),
                        "POS Hits Cumulative Pct": st.column_config.NumberColumn("POS Hits Cum %", format="%.0f%%"),
                        "POS Hits Score": st.column_config.NumberColumn("POS Hits Score", format="%.0f"),
                        "Combined Weighted Score": st.column_config.NumberColumn("Combined Weighted Score", format="%.2f"),
                        "Final Inventory Class": st.column_config.TextColumn("Final Inventory Class")
                    },
                    use_container_width=True
                )

                st.divider()

            # -----------------------------------------------------------------
            # BRANCH 2: Existing Inventory Valuation Analysis (Units & Unit Cost)
            # -----------------------------------------------------------------
            elif not missing_val_cols:
                # Execute deterministic pandas analysis
                analysis = analyze_inventory_dataframe(uploaded_df)
                processed_df = analysis["df"]

                st.header("📁 Uploaded Inventory Valuation & Analysis")

                # Display Dataset Dimensions & Column Names
                st.subheader("Dataset Dimensions & Schema")
                dim_col1, dim_col2, dim_col3 = st.columns([1, 1, 2])
                dim_col1.metric("Number of Rows", f"{analysis['rows']:,}")
                dim_col2.metric("Number of Columns", f"{analysis['columns']:,}")
                dim_col3.write(f"**Column Names**: {', '.join([f'`{c}`' for c in analysis['column_names']])}")

                st.write("")

                # Display Core Analytical Summary Results
                st.subheader("Key Inventory Analysis Results")
                kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                kpi1.metric("Total Units", f"{analysis['total_units']:,.0f}")
                kpi2.metric("Average Unit Cost", f"${analysis['avg_unit_cost']:,.2f}")
                kpi3.metric(
                    "Product with Highest Units",
                    analysis["product_highest_units"],
                    f"{analysis['highest_units_count']:,.0f} units"
                )
                kpi4.metric("Total Inventory Value", f"${analysis['total_inventory_value']:,.2f}")

                st.write("")

                # Category Grouping and Complete Processed Dataframe
                cat_col, table_col = st.columns([1, 2])
                with cat_col:
                    st.subheader("Total Units by Category")
                    st.dataframe(
                        analysis["total_units_by_category"],
                        column_config={
                            "Category": st.column_config.TextColumn("Category"),
                            "Total Units": st.column_config.NumberColumn("Total Units", format="%d")
                        },
                        hide_index=True,
                        use_container_width=True
                    )

                with table_col:
                    st.subheader("Uploaded Dataframe (with Inventory Value)")
                    st.dataframe(
                        processed_df,
                        column_config={
                            "Units": st.column_config.NumberColumn("Units", format="%d"),
                            "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                            "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f")
                        },
                        use_container_width=True
                    )

                st.divider()

            # -----------------------------------------------------------------
            # BRANCH 3: Schema Validation Error (Neither Schema Matched)
            # -----------------------------------------------------------------
            else:
                st.error(
                    f"❌ **Uploaded CSV does not match either supported inventory schema.**\n\n"
                    f"Please upload a CSV file conforming to one of the following schemas:\n\n"
                    f"1. **Multi-Criteria Inventory Classification Schema**:\n"
                    f"   - Required Columns: `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
                    f"   - Missing from your file: `{', '.join(missing_class_cols)}`\n\n"
                    f"2. **Inventory Valuation Schema**:\n"
                    f"   - Required Columns: `Product`, `Category`, `Units`, `Unit Cost`\n"
                    f"   - Missing from your file: `{', '.join(missing_val_cols)}`\n\n"
                    f"💡 *Tip: You can download formatted sample templates from the sidebar on the left.*"
                )

        except Exception as e:
            st.error(f"❌ Error processing uploaded CSV file: {str(e)}")
    else:
        # Guidance notice when no CSV is currently uploaded
        st.info(
            "💡 **CSV Inventory Data Upload & Analysis Available**:\n"
            "Use the **'Upload CSV File'** widget at the bottom of the left sidebar to upload inventory data. "
            "The system automatically detects your dataset format:\n"
            "- **Multi-Criteria Classification (A/B/C/D)**: requires `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
            "- **Inventory Valuation**: requires `Product`, `Category`, `Units`, `Unit Cost`\n\n"
            "*(Sample templates for both formats are available for instant download in the sidebar!)*"
        )

    # Main Tabs: Natural Language Agent vs. Interactive Calculators
    tab_agent, tab_calculators = st.tabs(["🤖 Natural Language Decision Agent", "🧮 Interactive Direct Calculators"])

    # -------------------------------------------------------------------------
    # TAB 1: Natural Language Decision Agent
    # -------------------------------------------------------------------------
    with tab_agent:
        st.subheader("Ask an Inventory Decision Question")
        st.write("Select a textbook benchmark scenario or enter your own business problem:")

        # Preset button short labels to prevent truncation
        PRESET_LABELS = {
            "1. Economic Order Quantity (EOQ)": "EOQ (Basic)",
            "2. Quantity Discount": "Quantity Discount",
            "3. Safety Stock & Reorder Point": "Safety Stock & Reorder Point",
            "4. Fixed-Period Review": "Fixed-Period Review",
            "5. Newsvendor (Single-Period)": "Newsvendor (Single-Period)"
        }

        # Initialize session state variables for active preset, query, and response
        if "active_preset" not in st.session_state:
            st.session_state.active_preset = "1. Economic Order Quantity (EOQ)"
        if "nl_query_box" not in st.session_state:
            st.session_state.nl_query_box = SAMPLE_QUERIES["1. Economic Order Quantity (EOQ)"]
        if "agent_response" not in st.session_state:
            st.session_state.agent_response = None

        def on_sample_button_clicked(preset_key: str, query_text: str):
            st.session_state.active_preset = preset_key
            st.session_state.nl_query_box = query_text
            st.session_state.agent_response = None  # Reset prior results until Analyze & Decide is clicked

        # If user manually modified query box away from the preset text, un-highlight preset
        if st.session_state.get("active_preset") and st.session_state.get("nl_query_box") != SAMPLE_QUERIES.get(st.session_state.active_preset):
            st.session_state.active_preset = None

        # 1-Click Example Buttons with balanced column width ratios
        cols = st.columns([1.0, 1.1, 1.4, 1.1, 1.3])
        for i, (label, sample_text) in enumerate(SAMPLE_QUERIES.items()):
            btn_label = PRESET_LABELS.get(label, label.split(". ")[-1])
            is_active = (st.session_state.get("active_preset") == label)
            btn_type = "primary" if is_active else "secondary"
            cols[i].button(
                btn_label,
                type=btn_type,
                use_container_width=True,
                key=f"sample_btn_{i}",
                on_click=on_sample_button_clicked,
                args=(label, sample_text)
            )

        user_query = st.text_area(
            "Natural Language Inventory Question:",
            height=95,
            key="nl_query_box"
        )

        col_btn, col_info = st.columns([1, 4])
        execute_click = col_btn.button("🚀 Analyze & Decide", type="primary", use_container_width=True)

        # Run Analysis ONLY when "Analyze & Decide" is clicked
        if execute_click:
            with st.spinner("Executing 7-stage inventory decision pipeline..."):
                st.session_state.agent_response = st.session_state.agent.process_query(user_query)

        response = st.session_state.agent_response

        # Display Results if a response is available
        if response is not None:
            st.divider()

            # Display Status & Model Header
            if response.success:
                st.success(f"**Identified Model**: {response.model_type.value} (Confidence: {response.confidence:.0%})")
            else:
                st.error(f"**Status**: {response.error_message}")

            # 7-Step Architecture Flow Display
            with st.expander("🔍 Architecture Pipeline Execution Trace (Click to inspect stages)", expanded=True):
                cols_step = st.columns(min(7, len(response.step_traces)))
                for idx, trace in enumerate(response.step_traces):
                    with cols_step[idx % len(cols_step)]:
                        st.markdown(f"**Stage {trace.step_number}**")
                        st.caption(trace.step_name)
                        st.info(trace.summary)

            if response.success and response.calculation_result:
                # Custom HTML/CSS Dark-Navy Executive Decision Recommendation Banner Card
                rec_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", response.executive_recommendation)
                st.markdown(
                    f"""<div style="background-color: #0f172a; border-radius: 10px; border-left: 5px solid #2563eb; padding: 18px 22px; margin-bottom: 20px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <div style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; color: #60a5fa; text-transform: uppercase;">
                                EXECUTIVE DECISION RECOMMENDATION
                            </div>
                            <div style="background-color: #1e293b; color: #93c5fd; font-size: 0.75rem; padding: 4px 12px; border-radius: 999px; white-space: nowrap;">
                                {response.model_type.value} • Conf: {response.confidence:.0%}
                            </div>
                        </div>
                        <div style="font-size: 1.05rem; font-weight: 500; color: #f8fafc; line-height: 1.5;">
                            {rec_html}
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

                st.write("")

                # KPI Metrics Cards
                st.subheader("Core Analytical Results")
                res = response.calculation_result
                m_type = response.model_type

                if m_type == InventoryModelType.EOQ:
                    k1, k2, k3, k4, k5, k6 = st.columns(6)
                    with k1:
                        st.markdown(
                            render_metric_card(
                                "Optimal EOQ",
                                f"{res['eoq_rounded']:,} units",
                                f"(exact: {res['eoq']:.1f})"
                            ),
                            unsafe_allow_html=True
                        )
                    with k2:
                        st.markdown(
                            render_metric_card(
                                "Order Frequency",
                                f"{res['orders_per_year']:.1f} orders/year"
                            ),
                            unsafe_allow_html=True
                        )
                    with k3:
                        st.markdown(
                            render_metric_card(
                                "Order Cycle Time",
                                f"{res['cycle_time_days']:.0f} work days",
                                f"({res['cycle_time_years']:.2f} yr)"
                            ),
                            unsafe_allow_html=True
                        )
                    with k4:
                        st.markdown(
                            render_metric_card(
                                "Annual Ordering Cost",
                                f"${res['annual_ordering_cost']:,.2f}"
                            ),
                            unsafe_allow_html=True
                        )
                    with k5:
                        st.markdown(
                            render_metric_card(
                                "Annual Holding Cost",
                                f"${res['annual_holding_cost']:,.2f}"
                            ),
                            unsafe_allow_html=True
                        )
                    with k6:
                        tot_val = f"${res['total_annual_cost']:,.2f}" if res.get("total_annual_cost") else f"${res['total_annual_inventory_cost']:,.2f}"
                        sub_var = f"(Variable: ${res['total_annual_inventory_cost']:,.2f})"
                        st.markdown(
                            render_metric_card(
                                "Total Annual Inv Cost",
                                tot_val,
                                sub_var
                            ),
                            unsafe_allow_html=True
                        )

                elif m_type == InventoryModelType.QUANTITY_DISCOUNT:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(render_metric_card("Optimal Tier", f"Tier {res['optimal_tier_index']}"), unsafe_allow_html=True)
                    with k2:
                        st.markdown(render_metric_card("Optimal Order Quantity", f"{res['optimal_order_quantity']:,.0f} units"), unsafe_allow_html=True)
                    with k3:
                        st.markdown(render_metric_card("Discounted Unit Price", f"${res['optimal_unit_price']:.2f}"), unsafe_allow_html=True)
                    with k4:
                        st.markdown(render_metric_card("Annual Cost Savings", f"${res['total_savings']:,.2f}", f"({res['savings_percentage']:.1f}% saved)"), unsafe_allow_html=True)

                elif m_type == InventoryModelType.SAFETY_STOCK_ROP:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(render_metric_card("Reorder Point (ROP)", f"{res['reorder_point_rounded']:,} units", f"(exact: {res['reorder_point']:.1f})"), unsafe_allow_html=True)
                    with k2:
                        st.markdown(render_metric_card("Safety Stock Buffer", f"{res['safety_stock_rounded']:,} units"), unsafe_allow_html=True)
                    with k3:
                        st.markdown(render_metric_card("Lead Time Demand", f"{res['expected_lead_time_demand']:,.1f} units"), unsafe_allow_html=True)
                    with k4:
                        st.markdown(render_metric_card("Cycle Service Level", f"{res['service_level']:.1%}", f"(Z = {res['z_score']:.2f})"), unsafe_allow_html=True)

                elif m_type == InventoryModelType.FIXED_PERIOD_REVIEW:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(render_metric_card("Recommended Order Q", f"{res['recommended_order_quantity_rounded']:,} units"), unsafe_allow_html=True)
                    with k2:
                        st.markdown(render_metric_card("Target Order-Up-To Level", f"{res['target_inventory_level_rounded']:,} units"), unsafe_allow_html=True)
                    with k3:
                        st.markdown(render_metric_card("Safety Stock", f"{res['safety_stock_rounded']:,} units"), unsafe_allow_html=True)
                    with k4:
                        st.markdown(render_metric_card("Protection Interval", f"{res['protection_interval']:.1f} periods", f"(P={res['review_period']} + L={res['lead_time']})"), unsafe_allow_html=True)

                elif m_type == InventoryModelType.NEWSVENDOR:
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.markdown(render_metric_card("Optimal Order Quantity", f"{res['optimal_order_quantity_rounded']:,} units", f"(exact: {res['optimal_order_quantity']:.1f})"), unsafe_allow_html=True)
                    with k2:
                        st.markdown(render_metric_card("Critical Fractile", f"{res['critical_fractile']:.2%}", f"(Cu=${res['underage_cost']} / Co=${res['overage_cost']})"), unsafe_allow_html=True)
                    with k3:
                        st.markdown(render_metric_card("Expected Maximum Profit", f"${res['expected_profit']:,.2f}"), unsafe_allow_html=True)
                    with k4:
                        st.markdown(render_metric_card("In-Stock Probability", f"{res['in_stock_probability']:.1%}"), unsafe_allow_html=True)

                # Verification & Sanity Checks
                st.subheader("Automated Mathematical & Operational Verification")
                for check in response.verification_checks:
                    if check.status == "PASS":
                        st.success(f"**{check.name} [PASSED]**: {check.message}")
                    elif check.status == "WARNING":
                        st.warning(f"**{check.name} [CAUTION]**: {check.message}")
                    else:
                        st.error(f"**{check.name} [FAILED]**: {check.message}")

                # Visual Charts
                if response.visualization_data:
                    st.subheader("Visual Optimization Curves")
                    vtype = response.visualization_data["type"]
                    vdata = response.visualization_data["data"]

                    if vtype == "eoq_curve":
                        df_chart = pd.DataFrame(vdata)
                        chart = alt.Chart(df_chart).transform_fold(
                            ["ordering_cost", "holding_cost", "total_inventory_cost"],
                            as_=["Cost Type", "Cost ($)"]
                        ).mark_line().encode(
                            x=alt.X("order_quantity:Q", title="Order Quantity (Units)"),
                            y=alt.Y("Cost ($):Q", title="Annual Cost ($)"),
                            color=alt.Color("Cost Type:N", title="Cost Component"),
                            tooltip=["order_quantity:Q", "Cost Type:N", "Cost ($):Q"]
                        ).properties(
                            title=f"EOQ Total Cost Trade-off Curve (Optimal Q* = {response.visualization_data['optimal_q']:.1f})",
                            height=360
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)

                    elif vtype == "newsvendor_curve":
                        df_chart = pd.DataFrame(vdata)
                        chart = alt.Chart(df_chart).mark_line(color="#1a73e8", strokeWidth=3).encode(
                            x=alt.X("order_quantity:Q", title="Order Quantity (Units)"),
                            y=alt.Y("expected_profit:Q", title="Expected Net Profit ($)"),
                            tooltip=["order_quantity:Q", "expected_profit:Q", "expected_sales:Q", "expected_leftover:Q"]
                        ).properties(
                            title=f"Newsvendor Expected Profit Curve (Optimal Q* = {response.visualization_data['optimal_q']:.1f})",
                            height=360
                        ).interactive()
                        st.altair_chart(chart, use_container_width=True)

                # Dedicated 8-Part Student Verification Breakdown for BAN 517
                if "inputs_extracted" in res or "intermediate_calculations" in res:
                    st.subheader("🎓 Student Verification Breakdown (8 Mandatory Sections)")
                    with st.expander("Click to expand/collapse interactive 8-Point Academic Verification tabs", expanded=True):
                        v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
                            "1 & 2: Inputs & Definitions",
                            "3 & 4: Formulas & Substitutions",
                            "5 & 6: Intermediate Steps & Final Answer",
                            "7 & 8: Verification & Assumptions"
                        ])
                        with v_tab1:
                            st.markdown("#### 1. Inputs Extracted")
                            st.json(res.get("inputs_extracted", {}))
                            st.markdown("#### 2. Variable Definitions")
                            for sym, desc in res.get("variable_definitions", {}).items():
                                st.markdown(f"- **{sym}**: {desc}")

                        with v_tab2:
                            st.markdown("#### 3. Formula")
                            form_dict = res.get("formula", {})
                            if isinstance(form_dict, dict):
                                for k, v in form_dict.items():
                                    st.markdown(f"- **{k.replace('_', ' ').title()}**: `{v}`")
                            else:
                                st.code(str(form_dict))

                            st.markdown("#### 4. Formula with Substituted Values")
                            sub_dict = res.get("formula_with_substituted_values", {})
                            if isinstance(sub_dict, dict):
                                for k, v in sub_dict.items():
                                    st.markdown(f"- **{k.replace('_', ' ').title()}**:\n  `{v}`")
                            else:
                                st.code(str(sub_dict))

                        with v_tab3:
                            st.markdown("#### 5. Intermediate Calculations")
                            inter = {k: v for k, v in res.get("intermediate_calculations", {}).items() if k != "tier_evaluations"}
                            if inter:
                                st.table(pd.DataFrame(list(inter.items()), columns=["Metric", "Value"]))
                            st.markdown("#### 6. Final Answer")
                            for k, v in res.get("final_answer", {}).items():
                                st.markdown(f"- **{k.replace('_', ' ').title()}**: **{v}**")

                        with v_tab4:
                            st.markdown("#### 7. Verification Checks")
                            for c in response.verification_checks:
                                st.markdown(f"- **[{c.status}] {c.name}**: {c.message}")
                            st.markdown("#### 8. Assumptions")
                            for i, a in enumerate(res.get("assumptions", []), 1):
                                st.markdown(f"{i}. {a}")

                # Detailed Explanation & KaTeX derivation
                st.subheader("Analytical Derivation & Managerial Insights")
                st.markdown(response.detailed_explanation)

    # -------------------------------------------------------------------------
    # TAB 2: Direct Calculators (What-If Analysis)
    # -------------------------------------------------------------------------
    with tab_calculators:
        st.subheader("Direct Parameter What-If Sensitivity Calculators")
        calc_choice = st.radio(
            "Select Inventory Model:",
            ["EOQ", "Quantity Discount", "Safety Stock & ROP", "Fixed-Period Review", "Newsvendor"],
            horizontal=True
        )

        if calc_choice == "EOQ":
            col1, col2, col3, col4 = st.columns(4)
            d = col1.number_input("Annual Demand (D)", min_value=1.0, value=10000.0, step=100.0)
            s = col2.number_input("Ordering / Setup Cost (S)", min_value=0.1, value=50.0, step=5.0)
            h = col3.number_input("Holding Cost / Unit / Year (H)", min_value=0.01, value=2.0, step=0.1)
            c = col4.number_input("Unit Purchase Price ($)", min_value=0.0, value=10.0, step=1.0)

            res = calculate_eoq(demand=d, ordering_cost=s, holding_cost=h, unit_cost=c)
            st.write(f"### Optimal EOQ: **{res.eoq_rounded:,} units** (Exact: {res.eoq:.2f})")
            st.write(f"Annual Inventory Cost: **${res.total_annual_inventory_cost:,.2f}** | Total Cost: **${res.total_annual_cost:,.2f}**")

        elif calc_choice == "Quantity Discount":
            col1, col2, col3 = st.columns(3)
            d = col1.number_input("Annual Demand (D)", min_value=1.0, value=1000.0, step=50.0)
            s = col2.number_input("Ordering Cost (S)", min_value=0.1, value=40.0, step=5.0)
            hr = col3.number_input("Holding Cost Rate (%)", min_value=1.0, max_value=100.0, value=20.0, step=1.0) / 100.0

            st.write("Default Discount Schedule: Tier 1 (1-499 @ $5.00), Tier 2 (500-999 @ $4.50), Tier 3 (1000+ @ $4.00)")
            brackets = [
                {"min_qty": 1, "max_qty": 499, "unit_price": 5.00},
                {"min_qty": 500, "max_qty": 999, "unit_price": 4.50},
                {"min_qty": 1000, "max_qty": None, "unit_price": 4.00}
            ]
            res = calculate_quantity_discount(demand=d, ordering_cost=s, holding_cost_rate=hr, discount_brackets=brackets)
            st.write(f"### Best Option: **Tier {res.optimal_tier_index}** ordering **{res.optimal_order_quantity:,.0f} units** @ ${res.optimal_unit_price:.2f}/unit")
            st.write(f"Total Annual Cost: **${res.optimal_total_annual_cost:,.2f}** | Savings: **${res.total_savings:,.2f} ({res.savings_percentage:.1f}%)**")

        elif calc_choice == "Safety Stock & ROP":
            col1, col2, col3, col4, col5 = st.columns(5)
            d = col1.number_input("Daily Demand Mean", min_value=0.1, value=100.0, step=5.0)
            sig_d = col2.number_input("Demand Std Dev", min_value=0.0, value=20.0, step=1.0)
            lt = col3.number_input("Lead Time (days)", min_value=0.1, value=4.0, step=1.0)
            sig_lt = col4.number_input("Lead Time Std Dev", min_value=0.0, value=0.0, step=0.5)
            sl = col5.slider("Service Level (%)", min_value=50.0, max_value=99.9, value=95.0, step=0.5) / 100.0

            res = calculate_safety_stock_rop(lead_time=lt, avg_demand=d, std_demand=sig_d, std_lead_time=sig_lt, service_level=sl)
            st.write(f"### Reorder Point: **{res.reorder_point_rounded:,} units** (Safety Stock: **{res.safety_stock_rounded:,} units**)")
            st.write(f"Expected Lead Time Demand: **{res.expected_lead_time_demand:.1f} units** | Z-score: **{res.z_score:.4f}**")

        elif calc_choice == "Fixed-Period Review":
            col1, col2, col3, col4 = st.columns(4)
            p = col1.number_input("Review Period P (days)", min_value=1.0, value=7.0, step=1.0)
            lt = col2.number_input("Lead Time L (days)", min_value=0.0, value=3.0, step=1.0)
            d = col3.number_input("Avg Daily Demand", min_value=0.1, value=50.0, step=5.0)
            sig_d = col4.number_input("Daily Demand Std Dev", min_value=0.0, value=10.0, step=1.0)

            col5, col6 = st.columns(2)
            on_hand = col5.number_input("Current On-Hand Inventory", min_value=0.0, value=150.0, step=10.0)
            on_order = col6.number_input("Pipeline On-Order Stock", min_value=0.0, value=100.0, step=10.0)

            res = calculate_fixed_period_review(
                review_period=p, lead_time=lt, avg_demand=d, std_demand=sig_d,
                service_level=0.95, current_inventory=on_hand, on_order=on_order
            )
            st.write(f"### Recommended Order Q: **{res.recommended_order_quantity_rounded:,} units**")
            st.write(f"Target Level (M): **{res.target_inventory_level_rounded:,} units** | Safety Stock: **{res.safety_stock_rounded:,} units**")

        elif calc_choice == "Newsvendor":
            col1, col2, col3 = st.columns(3)
            p = col1.number_input("Selling Price ($)", min_value=0.01, value=100.0, step=5.0)
            c = col2.number_input("Purchase Cost ($)", min_value=0.01, value=40.0, step=5.0)
            s = col3.number_input("Salvage Value ($)", min_value=0.0, value=10.0, step=5.0)

            col4, col5 = st.columns(2)
            mu = col4.number_input("Mean Demand", min_value=1.0, value=1000.0, step=50.0)
            sig = col5.number_input("Demand Std Dev", min_value=0.1, value=200.0, step=10.0)

            res = calculate_newsvendor(selling_price=p, purchase_cost=c, salvage_value=s, demand_mean=mu, demand_std=sig)
            st.write(f"### Optimal Order Quantity: **{res.optimal_order_quantity_rounded:,} units**")
            st.write(f"Critical Fractile: **{res.critical_fractile:.2%}** | Expected Profit: **${res.expected_profit:,.2f}**")


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    is_streamlit = False
    try:
        import streamlit.runtime
        if streamlit.runtime.exists():
            is_streamlit = True
    except Exception:
        pass

    if is_streamlit or ("streamlit" in sys.modules and any("streamlit" in str(arg).lower() for arg in sys.argv)):
        run_streamlit()
    elif len(sys.argv) > 1 and sys.argv[1].lower() in ["--cli", "-c"]:
        run_cli()
    else:
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                run_streamlit()
            else:
                run_cli()
        except Exception:
            run_cli()

