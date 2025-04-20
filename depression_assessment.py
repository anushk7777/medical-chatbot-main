import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


class PHQ9Assessment:
    def __init__(self):
        self.questions = [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling or staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
            "Trouble concentrating on things, such as reading the newspaper or watching television",
            "Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
            "Thoughts that you would be better off dead, or of hurting yourself in some way"
        ]

        self.options = {
            "Not at all": 0,
            "Several days": 1,
            "More than half the days": 2,
            "Nearly every day": 3
        }

    def display_questionnaire(self):
        """Display the PHQ-9 questionnaire and calculate score"""
        st.markdown("### PHQ-9 Depression Screening")
        st.markdown("""
        Over the last 2 weeks, how often have you been bothered by any of the following problems?
        """)

        responses = []
        for i, question in enumerate(self.questions):
            response = st.radio(
                f"{i + 1}. {question}",
                list(self.options.keys()),
                key=f"phq9_q{i}"
            )
            responses.append(self.options[response])

        if st.button("Calculate Score"):
            total_score = sum(responses)
            return total_score

        return None

    def interpret_score(self, score):
        """Interpret PHQ-9 score"""
        if score <= 4:
            return "Minimal or none", "You have few or no symptoms of depression."
        elif score <= 9:
            return "Mild", "You have some symptoms that may cause minor difficulties in daily life."
        elif score <= 14:
            return "Moderate", "Your symptoms are likely causing noticeable difficulties in daily functioning."
        elif score <= 19:
            return "Moderately severe", "Your symptoms are causing significant impairment in daily life and work."
        else:
            return "Severe", "Your symptoms are severe and significantly interfering with your ability to function."

    def get_recommendations(self, score):
        """Get recommendations based on PHQ-9 score"""
        recommendations = [
            "Regular physical activity (at least 30 minutes daily) can help improve mood",
            "Maintain a consistent sleep schedule",
            "Practice stress reduction techniques such as deep breathing or meditation"
        ]

        if score <= 4:
            recommendations.extend([
                "Continue healthy lifestyle habits",
                "Monitor your mood for any changes"
            ])
        elif score <= 9:
            recommendations.extend([
                "Consider self-help resources for mild depression",
                "Increase social connections and activities",
                "If symptoms persist for more than two weeks, consider consulting a healthcare provider"
            ])
        elif score <= 14:
            recommendations.extend([
                "Consider speaking with a healthcare provider about your symptoms",
                "Cognitive-behavioral techniques may be helpful",
                "Structure daily activities and set achievable goals",
                "Ensure adequate social support"
            ])
        elif score <= 19:
            recommendations.extend([
                "Speaking with a healthcare provider is strongly recommended",
                "Treatment options may include psychotherapy and possibly medication",
                "Create a support network of friends, family, or support groups",
                "Maintain regular daily routines"
            ])
        else:
            recommendations.extend([
                "Please consult with a healthcare provider promptly",
                "Treatment typically includes a combination of medication and psychotherapy",
                "If you have thoughts of harming yourself, call a crisis hotline or go to your nearest emergency room",
                "Having support from trusted friends or family members is important"
            ])

        return recommendations


class VisualizationTools:
    def create_phq9_gauge(self, score):
        """Create a gauge chart for PHQ-9 score"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Depression Severity"},
            gauge={
                'axis': {'range': [0, 27], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 4], 'color': '#92D050'},  # Minimal
                    {'range': [5, 9], 'color': '#FFC000'},  # Mild
                    {'range': [10, 14], 'color': '#FFA500'},  # Moderate
                    {'range': [15, 19], 'color': '#FF4500'},  # Moderately Severe
                    {'range': [20, 27], 'color': '#FF0000'}  # Severe
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))

        # Add annotations for severity levels
        fig.add_annotation(x=0.1, y=0.2, text="Minimal", showarrow=False)
        fig.add_annotation(x=0.3, y=0.2, text="Mild", showarrow=False)
        fig.add_annotation(x=0.5, y=0.2, text="Moderate", showarrow=False)
        fig.add_annotation(x=0.7, y=0.2, text="Mod. Severe", showarrow=False)
        fig.add_annotation(x=0.9, y=0.2, text="Severe", showarrow=False)

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=50),
        )

        return fig

    def create_phq9_history_chart(self, history):
        """Create a line chart of PHQ-9 score history"""
        dates = [item[0] for item in history]
        scores = [item[1] for item in history]

        df = pd.DataFrame({
            'Date': dates,
            'Score': scores
        })

        fig = px.line(df, x='Date', y='Score', markers=True)

        # Add horizontal bands for severity levels
        fig.add_hrect(y0=0, y1=4, line_width=0, fillcolor="#92D050", opacity=0.2)
        fig.add_hrect(y0=5, y1=9, line_width=0, fillcolor="#FFC000", opacity=0.2)
        fig.add_hrect(y0=10, y1=14, line_width=0, fillcolor="#FFA500", opacity=0.2)
        fig.add_hrect(y0=15, y1=19, line_width=0, fillcolor="#FF4500", opacity=0.2)
        fig.add_hrect(y0=20, y1=27, line_width=0, fillcolor="#FF0000", opacity=0.2)

        # Add severity labels
        fig.add_annotation(x=dates[-1], y=2, text="Minimal", showarrow=False, yshift=10)
        fig.add_annotation(x=dates[-1], y=7, text="Mild", showarrow=False, yshift=10)
        fig.add_annotation(x=dates[-1], y=12, text="Moderate", showarrow=False, yshift=10)
        fig.add_annotation(x=dates[-1], y=17, text="Mod. Severe", showarrow=False, yshift=10)
        fig.add_annotation(x=dates[-1], y=24, text="Severe", showarrow=False, yshift=10)

        fig.update_layout(
            title="PHQ-9 Score History",
            xaxis_title="Date",
            yaxis_title="Score",
            yaxis=dict(range=[0, 27]),
            height=350,
            margin=dict(l=20, r=20, t=50, b=50),
        )

        return fig

    def analyze_trend(self, history):
        """Analyze trend in PHQ-9 scores"""
        if len(history) < 2:
            return "Not enough data to analyze trend."

        scores = [item[1] for item in history]
        first_score = scores[0]
        last_score = scores[-1]

        if last_score < first_score - 5:
            return "Significant improvement in your symptoms over time."
        elif last_score < first_score - 2:
            return "Moderate improvement in your symptoms over time."
        elif last_score > first_score + 5:
            return "Significant worsening of symptoms over time. Consider consulting a healthcare provider."
        elif last_score > first_score + 2:
            return "Moderate worsening of symptoms over time. Monitor closely and consider professional support."
        else:
            return "Your symptoms have remained relatively stable over time."

    def create_risk_gauge(self, risk_score):
        """Create a gauge chart for risk score"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Level"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#92D050'},  # Low risk
                    {'range': [20, 50], 'color': '#FFC000'},  # Moderate risk
                    {'range': [50, 100], 'color': '#FF0000'},  # High risk
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))

        # Add annotations for risk levels
        fig.add_annotation(x=0.15, y=0.2, text="Low Risk", showarrow=False)
        fig.add_annotation(x=0.5, y=0.2, text="Moderate Risk", showarrow=False)
        fig.add_annotation(x=0.85, y=0.2, text="High Risk", showarrow=False)

        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=50),
        )

        return fig

    def create_factor_chart(self, factor_weights):
        """Create a bar chart of risk factors"""
        factors = list(factor_weights.keys())
        weights = list(factor_weights.values())

        # Create dataframe
        df = pd.DataFrame({
            'Factor': factors,
            'Weight': weights
        })

        # Sort by weight
        df = df.sort_values('Weight', ascending=False)

        # Create bar chart
        fig = px.bar(df, x='Weight', y='Factor', orientation='h',
                     labels={'Weight': 'Contribution to Risk', 'Factor': 'Risk Factor'},
                     color='Weight', color_continuous_scale=['#92D050', '#FFC000', '#FF0000'])

        fig.update_layout(
            title="Risk Factor Contribution",
            height=350,
            margin=dict(l=20, r=20, t=50, b=50),
        )

        return fig
