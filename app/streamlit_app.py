import streamlit as st
import requests

st.title("Mental Health Detection System")

st.write(
    "Enter a text sample to analyze mental health indicators."
)
text_input = st.text_area(
    "Enter Text",
    height=200
)

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter some text for analysis.")
    else:
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict",
                    json={"text": text_input}
                )
                response_data = response.json()
                if response.status_code == 200:
                    st.subheader("Prediction Result")
                    st.success(f"Predicted Class: {response_data['prediction']}")
                    # Display confidence scores as a bar chart
                    st.subheader("Confidence Scores")
                    st.bar_chart(
                        response_data["probabilities"]
                    )
                else:
                    st.error(f"Error: {response_data.get('error', 'Unknown error occurred.')}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray;'>

    Made with ❤️ by <b>Akashdeep Kashyap</b>

    <br>

    <a href="YOUR_GITHUB_LINK" target="_blank">
        GitHub Repository
    </a>

    </div>
    """,

    unsafe_allow_html=True
)