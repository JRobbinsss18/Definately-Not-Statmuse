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
        elif st.session_state.is_processing:
            self._display_loading_screen()
        else:
            self._display_main_interface()
    
    def _setup_page_config(self):
        st.markdown("""
        <style>
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding-top: 2rem;
        }
        
        .main-title {
            text-align: center;
            font-size: 3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1f2937;
        }
        
        .main-subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #6b7280;
            margin-bottom: 3rem;
        }
        
        .query-input {
            font-size: 1.1rem !important;
            border-radius: 8px !important;
            border: 1px solid #d1d5db !important;
            padding: 12px !important;
            margin-bottom: 1rem !important;
        }
        
        .example-queries {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin: 2rem 0;
        }
        
        .example-btn {
            background: #f3f4f6 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
            font-size: 0.9rem !important;
            color: #374151 !important;
        }
        
        .history-section {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #e5e7eb;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _display_main_interface(self):
        # Center content
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Main title
            st.markdown('<h1 class="main-title">Definately NOT SportsMuse</h1>', unsafe_allow_html=True)
            st.markdown('<p class="main-subtitle">Real Data and AI Fully Integrated</p>', unsafe_allow_html=True)
            
            # Main query UI
            user_query = st.text_input(
                "Ask your question",
                placeholder="Ask about any NBA player or team...",
                key="main_query_input",
                label_visibility="collapsed"
            )
            
            # Submit button
            if st.button("Ask", type="primary", use_container_width=True):
                if user_query.strip():
                    self._add_to_history(user_query)
                    self._process_query(user_query)
                else:
                    st.warning("Please enter a question!")
            
            # Example queries
            st.markdown('<div class="example-queries">', unsafe_allow_html=True)
            example_queries = [
                "Compare Lebron to Curry",
                "Analyze Klay Thompson",
                "Predict Coby White's stats over the next 4 years",
                "Bucks 2024 Season stats",
                "Warriors All time stats"
            ]
            
            # Sample displays
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
            
            # Query history
            if 'query_history' in st.session_state and st.session_state.query_history:
                st.markdown('<div class="history-section">', unsafe_allow_html=True)
                st.markdown("**Recent Questions:**")
                
                recent_queries = list(reversed(st.session_state.query_history[-3:]))
                for i, prev_query in enumerate(recent_queries):
                    # Truncate long queries
                    display_query = prev_query if len(prev_query) <= 50 else prev_query[:47] + "..."
                    if st.button(f"Recent: {display_query}", key=f"history_{i}", use_container_width=True):
                        self._process_query(prev_query)
                st.markdown('</div>', unsafe_allow_html=True)
    
    def _display_loading_screen(self):
        st.markdown("# Processing your query...")
        
        # Simple progress tracking
        progress_bar = st.progress(0)
        step_placeholder = st.empty()
        
        def update_progress(message, current_step, total_steps):
            progress = current_step / total_steps
            progress_bar.progress(progress)
            step_placeholder.markdown(f"{message} ({current_step}/{total_steps})")
        
        # Process query with progress updates
        try:
            results = self.query_router.process_query(
                st.session_state.current_query,
                progress_callback=update_progress
            )
            st.session_state.results = results
            st.session_state.is_processing = False
            st.rerun()
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            st.session_state.is_processing = False
            st.session_state.results = None
    
    def _display_results(self):
        results = st.session_state.results
        
        # Navigation and export buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("Back"):
                self._reset_to_main()
                return
        
        with col3:
            if st.button("Export PDF"):
                self._export_current_results_to_pdf()
        
        st.markdown(f"**Query:** {st.session_state.current_query}")
        
        # Display error if present
        if 'error' in results and results['error']:
            st.error(results["error"])
            return
        
        # Handle disambiguation cases
        if results.get('requires_disambiguation', False):
            self._display_disambiguation(results)
            return
        
        # Main Analysis
        if 'analysis' in results and results['analysis']:
            st.markdown("### Analysis")
            st.markdown(results['analysis'])
        
        # Prediction Summary (if available)
        if 'prediction_summary' in results:
            st.markdown("### ML Predictions")
            st.text(results['prediction_summary'])
        
        # Visualizations
        if 'visualizations' in results and results['visualizations']:
            st.markdown("### Charts & Analysis")
            
            for i, viz in enumerate(results['visualizations']):
                if 'title' in viz:
                    st.markdown(f"**{viz['title']}**")
                
                if 'chart' in viz and viz['chart']:
                    try:
                        st.plotly_chart(viz['chart'], use_container_width=True)
                        
                        # Add individual chart analysis
                        chart_analysis = self._generate_chart_analysis(viz, results, i)
                        if chart_analysis:
                            st.markdown(chart_analysis)
                            st.markdown("---")
                        
                    except Exception as e:
                        st.error(f"Chart error: {str(e)}")
        
        # Raw Data (expandable)
        if 'data' in results and results['data']:
            with st.expander("View Raw Data"):
                for key, value in results['data'].items():
                    if hasattr(value, 'to_dict'):  # DataFrame
                        st.subheader(key.replace('_', ' ').title())
                        st.dataframe(value)
                    elif isinstance(value, dict):
                        st.json(value)
                    else:
                        st.text(f"{key}: {value}")
        
        # Simple action button
        if st.button("Ask Another Question", type="primary"):
            self._reset_to_main()
    
    def _add_to_history(self, query: str):
        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
            if len(st.session_state.query_history) > 10:
                st.session_state.query_history.pop(0)
    
    def _process_query(self, query: str):
        self.query_router.nba_client.clear_disambiguation()
        
        st.session_state.current_query = query
        st.session_state.is_processing = True
        st.session_state.results = None
        st.rerun()
    
    def _reset_to_main(self):
        self.query_router.nba_client.clear_disambiguation()
        
        st.session_state.current_query = ""
        st.session_state.is_processing = False
        st.session_state.results = None
    
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
        
        # Create columns for side-by-side buttons
        cols = st.columns(len(options))
        
        for i, (col, option) in enumerate(zip(cols, options)):
            with col:
                st.markdown(f"""
                <div style="
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 16px; 
                    text-align: center;
                    margin: 8px 0;
                    background: white;
                ">
                    <h4 style="margin: 0 0 8px 0;">{option['name']}</h4>
                    <p style="margin: 0; color: #333; font-size: 14px; font-weight: 500;">Confidence: {option['confidence']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Button to select this player
                if st.button(f"Select {option['name']}", key=f"select_{option['id']}", use_container_width=True):
                    # Reprocess the query with the full player name
                    original_query = st.session_state.current_query
                    if query in original_query:
                        new_query = original_query.replace(query, option['name'])
                    else:
                        # Fallback: append the full name
                        new_query = f"{original_query.split()[0]} {option['name']}"
                    self._process_query(new_query)
        
        st.markdown("---")
        st.markdown("*Or you can go back and try your query again with the full player name.*")
        
        if st.button("Back to Search", type="secondary"):
            self._reset_to_main()
    
    def _create_sample_chart(self):
        fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
        fig.update_layout(title="Sample Chart")
        return fig