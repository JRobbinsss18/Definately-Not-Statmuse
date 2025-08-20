import streamlit as st
import time
from typing import Dict, Any
from src.query_router import QueryRouter
import plotly.graph_objects as go
class NBASportsMuseDashboard:
    def __init__(self):
        self.query_router = QueryRouter()
        self._initialize_session_state()
    def _initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        if 'is_processing' not in st.session_state:
            st.session_state.is_processing = False
        if 'results' not in st.session_state:
            st.session_state.results = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
    def run(self):
        self._setup_page_config()
        if st.session_state.results:
            self._display_results()
        else:
            self._display_main_interface()
    def _setup_page_config(self):
        pass
    def _display_main_interface(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="main-title">Definately NOT SportsMuse</h1>', unsafe_allow_html=True)
            st.markdown('<p class="main-subtitle">Real Data and AI Fully Integrated</p>', unsafe_allow_html=True)
            user_query = st.text_input(
                "Ask your question",
                placeholder="Ask about any NBA player or team...",
                key="main_query_input",
                label_visibility="collapsed"
            )
            if st.button("Ask", type="primary", use_container_width=True, key="main_ask_button"):
                if user_query.strip():
                    self._add_to_history(user_query)
                    self._process_query(user_query)
                else:
                    st.warning("Please enter a question!")
            st.markdown('<div class="example-queries">', unsafe_allow_html=True)
            example_queries = [
                "Compare Lebron to Curry",
                "Analyze Klay Thompson",
                "Predict Coby White's stats over the next 4 years",
                "Bucks 2024 Season stats",
                "Warriors All time stats"
            ]
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(example_queries[0], key="example_0", use_container_width=True):
                    self._add_to_history(example_queries[0])
                    self._process_query(example_queries[0])
            with col2:
                if st.button(example_queries[1], key="example_1", use_container_width=True):
                    self._add_to_history(example_queries[1])
                    self._process_query(example_queries[1])
            with col3:
                if st.button(example_queries[2], key="example_2", use_container_width=True):
                    self._add_to_history(example_queries[2])
                    self._process_query(example_queries[2])
            col4, col5, col6 = st.columns([1, 2, 1])
            with col5:
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button(example_queries[3], key="example_3", use_container_width=True):
                        self._add_to_history(example_queries[3])
                        self._process_query(example_queries[3])
                with col_b:
                    if st.button(example_queries[4], key="example_4", use_container_width=True):
                        self._add_to_history(example_queries[4])
                        self._process_query(example_queries[4])
            st.markdown('</div>', unsafe_allow_html=True)
            if 'query_history' in st.session_state and st.session_state.query_history:
                st.markdown('<div class="history-section">', unsafe_allow_html=True)
                st.markdown("**Recent Questions:**")
                recent_queries = list(reversed(st.session_state.query_history[-3:]))
                for i, prev_query in enumerate(recent_queries):
                    display_query = prev_query if len(prev_query) <= 50 else prev_query[:47] + "..."
                    if st.button(f"Recent: {display_query}", key=f"history_{i}", use_container_width=True):
                        self._process_query(prev_query)
                st.markdown('</div>', unsafe_allow_html=True)
    def _display_results(self):
        results = st.session_state.results
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("Back", key="back_to_main_button"):
                self._reset_to_main()
                return
        with col3:
            if st.button("Export PDF"):
                self._export_current_results_to_pdf()
        st.markdown(f"**Query:** {st.session_state.current_query}")
        if 'error' in results and results['error']:
            st.error(results["error"])
            return
        if results.get('requires_disambiguation', False):
            self._display_disambiguation(results)
            return
        if 'analysis' in results and results['analysis']:
            st.markdown("### Analysis")
            st.markdown(results['analysis'])
        if 'prediction_summary' in results:
            st.markdown("### ML Predictions")
            st.text(results['prediction_summary'])
        if 'visualizations' in results and results['visualizations']:
            st.markdown("### Charts & Analysis")
            for i, viz in enumerate(results['visualizations']):
                if 'title' in viz:
                    st.markdown(f"**{viz['title']}**")
                if 'chart' in viz and viz['chart']:
                    try:
                        st.plotly_chart(viz['chart'], use_container_width=True)
                        chart_analysis = self._generate_chart_analysis(viz, results, i)
                        if chart_analysis:
                            st.markdown(chart_analysis)
                            st.markdown("---")
                    except Exception as e:
                        st.error(f"Chart error: {str(e)}")
        if 'data' in results and results['data']:
            with st.expander("View Raw Data"):
                for key, value in results['data'].items():
                    if hasattr(value, 'to_dict'):
                        st.subheader(key.replace('_', ' ').title())
                        st.dataframe(value)
                    elif isinstance(value, dict):
                        st.json(value)
                    else:
                        st.text(f"{key}: {value}")
        if st.button("Ask Another Question", type="primary", key="ask_another_button"):
            self._reset_to_main()
    def _add_to_history(self, query: str):
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
            if len(st.session_state.query_history) > 10:
                st.session_state.query_history.pop(0)
    def _process_query(self, query: str):
        self.query_router.nba_client.clear_disambiguation()
        st.session_state.current_query = query
        st.session_state.is_processing = False
        st.session_state.results = None
        self._execute_query_with_progress(query)
    def _execute_query_with_progress(self, query: str):
        st.markdown("# Processing your query...")
        progress_bar = st.progress(0)
        step_placeholder = st.empty()
        def update_progress(message, current_step, total_steps):
            progress = current_step / total_steps
            progress_bar.progress(progress)
            step_placeholder.markdown(f"{message} ({current_step}/{total_steps})")
        try:
            results = self.query_router.process_query(
                query,
                progress_callback=update_progress
            )
            st.session_state.results = results
            st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.session_state.results = None
    def _reset_to_main(self):
        self.query_router.nba_client.clear_disambiguation()
        st.session_state.current_query = ""
        st.session_state.is_processing = False
        st.session_state.results = None
        st.rerun()
    def _export_current_results_to_pdf(self):
        try:
            results = st.session_state.results
            if not results:
                st.error("No results to export")
                return
            with st.spinner("Generating PDF report..."):
                pdf_path = self.query_router.export_to_pdf(results)
            st.success(f"PDF report generated successfully!")
            st.info(f"Saved to: {pdf_path}")
            try:
                with open(pdf_path, "rb") as pdf_file:
                    pdf_data = pdf_file.read()
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=f"NBA_Report_{st.session_state.current_query[:30].replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
            except Exception as download_error:
                st.warning(f"PDF created but download failed: {download_error}")
        except Exception as e:
            st.error(f"Failed to export PDF: {str(e)}")
    def _generate_chart_analysis(self, viz, results, chart_index):
        if not viz.get('title'):
            return None
        title = viz['title'].lower()
        if 'radar' in title or 'comparison' in title:
            return "*Radar chart compares overall statistical performance. Larger areas indicate superior performance.*"
        elif 'prediction' in title:
            return "*ML prediction based on historical patterns and career trajectory.*"
        elif 'progression' in title:
            return "*Career progression showing performance trends over time.*"
        elif 'distribution' in title:
            return "*Statistical distribution showing consistency and variability.*"
        return None
    def _display_disambiguation(self, results):
        st.markdown("### Multiple Players Found")
        disambiguation_data = results['data']['disambiguation_options']
        query = disambiguation_data['query']
        options = disambiguation_data['options']
        st.markdown(f"We found multiple players for **'{query}'**. Which player did you mean?")
        st.markdown("---")
        cols = st.columns(len(options))
        for i, (col, option) in enumerate(zip(cols, options)):
            with col:
                st.markdown(f"**{option['name']}**", unsafe_allow_html=True)
                if st.button(f"Select {option['name']}", key=f"select_{option['id']}", use_container_width=True):
                    original_query = st.session_state.current_query
                    if query in original_query:
                        new_query = original_query.replace(query, option['name'])
                    else:
                        new_query = f"{original_query.split()[0]} {option['name']}"
                    self._process_query(new_query)
        st.markdown("---")
        st.markdown("*Or you can go back and try your query again with the full player name.*")
        if st.button("Back to Search", type="secondary", key="back_to_search_button"):
            self._reset_to_main()
    def _create_sample_chart(self):
        fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
        fig.update_layout(title="Sample Chart")
        return fig