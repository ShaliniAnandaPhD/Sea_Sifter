# Sea_Sifter (Systematic Identification of Floating Trash and Environmental Remediation)

Seasifter harnesses MongoDB to store open-source microplastic contamination data. By combining the latest research on plastic removal innovations, Seasifter enables interactive mapping of accumulation hotspots with NLP-powered clean-up recommendations tailored to each location's plastic density.

🌊 **Seasifter: A Leap Forward in Tackling Microplastics Pollution** 🚀

It's been 4 months since the MongoDB hackathon, and Seasifter has evolved significantly. We are excited to share the latest updates and features of our tool designed to visualize and analyze global microplastic pollution, now on track to become an interactive chatbot!

## 🔍 Seasifter's Cutting-Edge Features

- **Dynamic Mapping with Folium**: Explore an interactive map showcasing microplastic accumulation across oceans and coastlines.
- **Analytics**: Use Auto-Regressive Integrated Moving Average (ARIMA) to predict microplastic accumulation over 1, 2, 5, and 10 years.
- **Scientific Insights with LangChain**: Seamlessly synthesize and summarize scientific research, focusing on plastic remediation methods and studies.

## 📊 Data-Driven Approach

- **MongoDB Database**: Powers our core visualizations with extensive microplastic contamination data.
- **LangChain Integration**: Enhances our capability to process and understand vast amounts of scientific literature, turning complex data into actionable insights.

## 🌐 Future Vision: Seasifter Chatbot

- **Interactive Chatbot**: Evolve Seasifter into an interactive chatbot, making environmental data more accessible and engaging.
- **Targeted Insights**: Identify areas like San Domingo in the Dominican Republic, predicted to experience a surge in microplastic accumulation. Targeted insights are critical for effective clean-up operations.

## 🖥️ Tech Stack

- **MongoDB**: For robust data management.
- **Folium**: For creating intuitive, interactive maps.
- **LangChain**: For synthesizing scientific papers and generating knowledge graphs.
- **Rasa**: For developing the chatbot interface.
- **Python Libraries**: For data ingestion, processing, and visualization.

## 📂 Latest Files and Modules

1. `research_synthesizer.py`: Synthesizes research papers to extract actionable insights on plastic remediation.
2. `policy_recommendation_generator.py`: Generates policy recommendations based on synthesized research data.
3. `knowledge_graph_builder.py`: Builds knowledge graphs to visually represent relationships between different entities in the research data.
4. `impact_assessment.py`: Assesses the potential impact of different remediation strategies.
5. `entity_extraction.py`: Extracts relevant entities from research papers for further analysis.
6. `document_qa.py`: Provides a question-answering interface for querying the synthesized research.
7. `data_ingestion_pipeline.py`: Manages the ingestion of microplastic contamination data into MongoDB.
8. `custom_prompt_openai.py`: Customizes prompts for OpenAI's language model to generate relevant responses.
9. `cleanup_recommendation_engine.py`: Recommends specific clean-up strategies based on contamination data.
10. `chatbot_interface.py`: Interface for the Seasifter chatbot, facilitating user interactions with the system.
11. `text_generation.py`: Generates text outputs for various components of the system, enhancing user interaction and report generation.
12. `microplastic_trajectory_prediction.py`: Predicts the future movement of microplastics in the ocean.
13. `microplastic_source_identifier.py`: Identifies the sources of microplastic pollution.
14. `microplastic_policy_simulator.py`: Simulates the impact of different policies on microplastic pollution levels.
15. `microplastic_ingestion_risk.py`: Assesses the risk of microplastic ingestion by marine life.
16. `microplastic_image_analysis.py`: Analyzes images to detect and quantify microplastic particles.
17. `microplastic_biodegradation_simulator.py`: Simulates the biodegradation process of microplastics.
18. `citizen_science_gamification.py`: Encourages public participation through gamification, collecting data on microplastic pollution.

## 🗂️ Data Sources

- **Scientific Literature**: Research papers on microplastic pollution and remediation.
- **Environmental Databases**: Datasets from environmental monitoring agencies.
- **Satellite Imagery**: High-resolution images for mapping microplastic distribution.
- **Crowdsourced Data**: Data collected from citizen scientists and environmental organizations.

## 🤖 AI/ML Tools Used 

### Natural Language Processing (NLP)
- **Modules**: `research_synthesizer.py`, `entity_extraction.py`
- **Function**: Processes and extracts information from scientific papers, allowing for comprehensive data synthesis and summarization.

### Predictive Modeling
- **Modules**: `microplastic_trajectory_prediction.py`
- **Function**: Uses ARIMA models to forecast the future movement of microplastics in oceans, providing critical data for preventive measures.

### Image Analysis
- **Modules**: `microplastic_image_analysis.py`
- **Function**: Detects and quantifies microplastics in environmental samples using advanced image processing techniques.

### Policy Simulation
- **Modules**: `microplastic_policy_simulator.py`
- **Function**: Assesses the impact of different environmental policies on microplastic pollution levels, aiding in strategic planning.

### Knowledge Graphs
- **Modules**: `knowledge_graph_builder.py`
- **Function**: Represents relationships between various entities in the research data, facilitating a deeper understanding of complex interactions.

### Chatbot Development
- **Modules**: `chatbot_interface.py`, `rasa`
- **Function**: Creates an interactive, user-friendly chatbot that provides real-time information and recommendations.

## 🔗 GitHub Repository

[Seasifter GitHub Repository](https://github.com/ShaliniAnandaPhD/Sea_Sifter)

## 📜 Getting Started

### Prerequisites

- Python 3.8 or higher
- MongoDB
- An API key for OpenAI (for text generation and NLP tasks)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShaliniAnandaPhD/Sea_Sifter.git
   cd Sea_Sifter
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables (e.g., MongoDB connection string, OpenAI API key) in a `.env` file:
   ```env
   MONGODB_URI="your_mongodb_uri"
   OPENAI_API_KEY="your_openai_api_key"
   ```

### Running the Application

1. Start the data ingestion pipeline to populate the MongoDB database:
   ```bash
   python data_ingestion_pipeline.py
   ```

2. Run the research synthesizer to process and summarize scientific literature:
   ```bash
   python research_synthesizer.py
   ```

3. Generate policy recommendations:
   ```bash
   python policy_recommendation_generator.py
   ```

4. Build the knowledge graph:
   ```bash
   python knowledge_graph_builder.py
   ```

5. Assess the impact of remediation strategies:
   ```bash
   python impact_assessment.py
   ```

6. Extract relevant entities from research papers:
   ```bash
   python entity_extraction.py
   ```

7. Set up the chatbot interface:
   ```bash
   rasa train
   rasa run
   ```

8. Explore the interactive map:
   ```bash
   python visualize_map.py
   ```

## 📜 Requirements

The project dependencies are listed in `requirements.txt`, which includes core libraries such as:
- **pymongo**: For MongoDB interactions.
- **langchain**: For integrating and processing scientific literature.
- **rasa**: For developing the chatbot interface.
- **folium**: For creating interactive maps.
- **nltk**: For natural language processing tasks.
- **transformers**: For advanced text processing and generation.
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **scikit-learn**: For machine learning algorithms and models.
- **beautifulsoup4**: For web scraping and data ingestion.
- **requests**: For making HTTP requests.

---

Feel free to further refine or customize any specific sections as needed!

GitHub: https://lnkd.in/gPt322_K

Next Steps:
Turn this into a chatbot

📸 [Attached Image: Microplastic Accumulation Visualization near San Domingo] - The red zones are areas of high microplastic content. You can see San Domingo in Dominical Republic is fairly low as of today but will turn into a red zone in 2 years.
![Screenshot 2024-01-28 at 7 56 37 AM](https://github.com/ShaliniAnandaPhD/Sea_Sifter/assets/50239203/aedd7634-1aec-4a46-81af-1237786d40c6)

![Screenshot 2024-01-28 at 7 55 52 AM](https://github.com/ShaliniAnandaPhD/Sea_Sifter/assets/50239203/236aeb7b-ec4f-42a2-817e-fe692122b1f0)


