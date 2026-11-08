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
    analyze_inventory_dataframe
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

    # -------------------------------------------------------------------------
    # Uploaded CSV Data Processing and Analysis
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            missing_cols = validate_inventory_csv_columns(uploaded_df)

            if missing_cols:
                st.error(
                    f"❌ **Missing Required Columns**: `{', '.join(missing_cols)}`.\n\n"
                    f"The uploaded CSV must contain the following required columns: "
                    f"`Product`, `Category`, `Units`, `Unit Cost`."
                )
            else:
                # Execute deterministic pandas analysis
                analysis = analyze_inventory_dataframe(uploaded_df)
                processed_df = analysis["df"]

                st.header("📁 Uploaded Inventory Data & Analysis")

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

        except Exception as e:
            st.error(f"❌ Error processing uploaded CSV file: {str(e)}")

    # Main Tabs: Natural Language Agent vs. Interactive Calculators
    tab_agent, tab_calculators = st.tabs(["🤖 Natural Language Decision Agent", "🧮 Interactive Direct Calculators"])

    # -------------------------------------------------------------------------
    # TAB 1: Natural Language Decision Agent
    # -------------------------------------------------------------------------
    with tab_agent:
        st.subheader("1. Ask an Inventory Decision Question")
        st.write("Select a textbook benchmark scenario or enter your own business problem:")

        # 1-Click Example Buttons
        cols = st.columns(len(SAMPLE_QUERIES))
        selected_sample = None
        for i, (label, sample_text) in enumerate(SAMPLE_QUERIES.items()):
            short_name = label.split(". ")[1]
            if cols[i].button(short_name, use_container_width=True):
                selected_sample = sample_text

        default_query = selected_sample if selected_sample else SAMPLE_QUERIES["1. Economic Order Quantity (EOQ)"]

        user_query = st.text_area(
            "Natural Language Inventory Question:",
            value=default_query,
            height=95,
            key="nl_query_box"
        )

        col_btn, col_info = st.columns([1, 4])
        execute_click = col_btn.button("🚀 Analyze & Decide", type="primary", use_container_width=True)

        # Run Analysis
        if execute_click or selected_sample:
            with st.spinner("Executing 7-stage inventory decision pipeline..."):
                response: AgentResponse = st.session_state.agent.process_query(user_query)

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
                # Executive Recommendation
                st.subheader("2. Executive Recommendation")
                st.markdown(
                    f"""<div style="background-color: #e8f0fe; border-left: 6px solid #1a73e8; padding: 16px 20px; border-radius: 6px; font-size: 1.05em; color: #174ea6;">
                    {response.executive_recommendation}
                    </div>""",
                    unsafe_allow_html=True
                )

                st.write("")

                # KPI Metrics Cards
                st.subheader("3. Core Analytical Results")
                res = response.calculation_result
                m_type = response.model_type

                if m_type == InventoryModelType.EOQ:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Optimal EOQ", f"{res['eoq_rounded']:,} units", f"Exact: {res['eoq']:.1f}")
                    k2.metric("Total Annual Inv Cost", f"${res['total_annual_inventory_cost']:,.2f}")
                    k3.metric("Order Frequency", f"{res['orders_per_year']:.1f} orders/yr")
                    k4.metric("Order Cycle Time", f"{res['cycle_time_days']:.0f} work days")

                elif m_type == InventoryModelType.QUANTITY_DISCOUNT:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Optimal Tier", f"Tier {res['optimal_tier_index']}")
                    k2.metric("Optimal Order Quantity", f"{res['optimal_order_quantity']:,.0f} units")
                    k3.metric("Discounted Unit Price", f"${res['optimal_unit_price']:.2f}")
                    k4.metric("Annual Cost Savings", f"${res['total_savings']:,.2f}", f"{res['savings_percentage']:.1f}% saved")

                elif m_type == InventoryModelType.SAFETY_STOCK_ROP:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Reorder Point (ROP)", f"{res['reorder_point_rounded']:,} units", f"Exact: {res['reorder_point']:.1f}")
                    k2.metric("Safety Stock Buffer", f"{res['safety_stock_rounded']:,} units")
                    k3.metric("Lead Time Demand", f"{res['expected_lead_time_demand']:,.1f} units")
                    k4.metric("Cycle Service Level", f"{res['service_level']:.1%}", f"Z = {res['z_score']:.2f}")

                elif m_type == InventoryModelType.FIXED_PERIOD_REVIEW:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Recommended Order Q", f"{res['recommended_order_quantity_rounded']:,} units")
                    k2.metric("Target Order-Up-To Level", f"{res['target_inventory_level_rounded']:,} units")
                    k3.metric("Safety Stock", f"{res['safety_stock_rounded']:,} units")
                    k4.metric("Protection Interval", f"{res['protection_interval']:.1f} periods", f"P={res['review_period']} + L={res['lead_time']}")

                elif m_type == InventoryModelType.NEWSVENDOR:
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Optimal Order Quantity", f"{res['optimal_order_quantity_rounded']:,} units", f"Exact: {res['optimal_order_quantity']:.1f}")
                    k2.metric("Critical Fractile", f"{res['critical_fractile']:.2%}", f"Cu=${res['underage_cost']} / Co=${res['overage_cost']}")
                    k3.metric("Expected Maximum Profit", f"${res['expected_profit']:,.2f}")
                    k4.metric("In-Stock Probability", f"{res['in_stock_probability']:.1%}")

                # Verification & Sanity Checks
                st.subheader("4. Automated Mathematical & Operational Verification")
                for check in response.verification_checks:
                    if check.status == "PASS":
                        st.success(f"**{check.name} [PASSED]**: {check.message}")
                    elif check.status == "WARNING":
                        st.warning(f"**{check.name} [CAUTION]**: {check.message}")
                    else:
                        st.error(f"**{check.name} [FAILED]**: {check.message}")

                # Visual Charts
                if response.visualization_data:
                    st.subheader("5. Visual Optimization Curves")
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
                    st.subheader("6. 🎓 Student Verification Breakdown (8 Mandatory Sections)")
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
                st.subheader("7. Analytical Derivation & Managerial Insights")
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
    # If run through `streamlit run app.py`, 'streamlit' is in sys.modules and runs streamlit
    # If run through `python app.py`, provide CLI interface
    if "streamlit" in sys.modules and any("streamlit" in arg for arg in sys.argv):
        run_streamlit()
    else:
        # Check if launched via `streamlit run`
        try:
            import streamlit.runtime
            if streamlit.runtime.exists():
                run_streamlit()
            else:
                run_cli()
        except ImportError:
            run_cli()

