# Customer Review Engine

This project automates the process of harvesting and summarizing Amazon customer reviews for a specified product. Using OpenAI, it provides a concise summary of customer feedback, helping users quickly understand the general sentiment and key points from multiple reviews.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/icoshikhar/cyborgs
   ```

2. 


### Configuration

Create a `.env` file and add your own OpenAI API key in the `.env` file as follows:

```bash
GEMINI_API_KEY='your-key-here'
```

### Usage

1. After installing the dependencies, you can run the Streamlit app in root directory by executing the following command:
   
   ```bash
   streamlit run app.py
   ```

2. Follow the prompts to input the product description 

3. Move the slide bar to select number of reviews to pull.

4. The script will display the top 10 ranking customer reviews and the overall review summary.