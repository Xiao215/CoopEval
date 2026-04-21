# LLM Judge Package

A flexible framework for using Large Language Models to classify and analyze text data according to custom taxonomies.

## 🌟 Features

- **Custom Taxonomies**: Define your own classification categories and descriptions
- **Hosted API Support**: Works with OpenRouter or any OpenAI-compatible client
- **Batch Processing**: Efficiently process thousands of texts with parallel execution
- **Rate Limiting**: Built-in rate limiting and error handling
- **Resumability**: Process large datasets with interruption recovery
- **Cost Tracking**: Monitor API usage and costs
- **Flexible Output**: JSON-structured results with confidence scores

## 🚀 Quick Start

### 1. Installation

```bash
# From the repository root, install the package
pip install -e .
```

### 2. Set up API credentials

Choose one of the supported providers and set environment variables:

**OpenRouter:**

```bash
export OPENROUTER_API_KEY="your-api-key"
export OPENROUTER_MODEL_NAME="openai/gpt-4o-mini"  # or your preferred model
```

### 3. Basic Usage

```python
import sys
sys.path.append('llm_judge_package/src')

from api_clients import OpenRouterClient
from taxonomy import Taxonomy
from judge import LLMJudge

# Set up API client
client = OpenRouterClient(
    api_key="your-key",
    model_name="openai/gpt-4o-mini",
)

# Load taxonomy
taxonomy = Taxonomy.from_json_file('llm_judge_package/config/example_taxonomy.json')

# Create judge
judge = LLMJudge(client, taxonomy)

# Classify text
result = judge.classify_text("I'm really happy with this product!")
print(f"Classification: {result['justification_type']}")
print(f"Confidence: {result['Confidence']}")
```

## 📋 Creating Custom Taxonomies

### Method 1: JSON File

Create a JSON file with your categories:

```json
{
  "categories": {
    "Technical Issue": "Customer is reporting a bug, error, or technical problem with the product.",
    "Feature Request": "Customer is suggesting new features or improvements.",
    "Billing Question": "Customer has questions about pricing, charges, or billing.",
    "General Inquiry": "Customer is asking for general information or clarification."
  }
}
```

### Method 2: Programmatically

```python
from taxonomy import Taxonomy

# Create empty taxonomy
taxonomy = Taxonomy()

# Add categories
taxonomy.add_category(
    "Positive Feedback",
    "Customer expressing satisfaction or praise"
)
taxonomy.add_category(
    "Negative Feedback",
    "Customer expressing dissatisfaction or complaints"
)

# Save for later use
taxonomy.save_to_json('my_taxonomy.json')
```

## 🔄 Batch Processing

For processing large datasets:

```python
import pandas as pd
from processors import BatchProcessor

# Load your data
df = pd.DataFrame({
    'text': ['Text 1', 'Text 2', 'Text 3'],
    'user_id': ['user1', 'user2', 'user3']
})

# Create batch processor
batch_processor = BatchProcessor(judge, max_workers=4, batch_size=10)

# Process dataframe
df_results = batch_processor.process_dataframe(
    df,
    text_column='text',
    metadata_columns=['user_id']
)

# Results are added as new columns
print(df_results[['text', 'classification_justification', 'classification_confidence']])
```

## 📊 Output Format

Each classification returns a dictionary with:

```python
{
    "classification_explanation": "Detailed reasoning for the classification",
    "classification_confidence": 0.85,  # Float between 0 and 1
    "classification_justification": "Category1, Category2"  # Can be multiple categories
}
```

## 🛠️ API Clients

### OpenRouter

```python
from api_clients import OpenRouterClient

client = OpenRouterClient(
    api_key="your-api-key",
    model_name="openai/gpt-4o-mini",
    site_url="https://your-site.com",  # Optional
    site_name="Your App Name"  # Optional
)
```

## 📁 Project Structure

```bash
llm_judge_package/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── judge.py             # Core LLM judge functionality
│   ├── api_clients.py       # API client implementations
│   ├── taxonomy.py          # Taxonomy management
│   └── processors.py        # Batch processing utilities
├── config/
│   ├── cooperation_taxonomy.json    # Example: cooperation strategies taxonomy
│   └── example_taxonomy.json       # Example: general sentiment taxonomy
├── examples/
│   └── basic_usage.py              # Basic usage example
├── tests/                          # Unit tests (future)
├── pyproject.toml                  # Project metadata and Python dependencies
└── README.md                       # This file
```

## 🎯 Examples

### Example 1: Basic Usage

See `examples/basic_usage.py` for a complete example showing:

- API client setup
- Taxonomy loading
- Single text classification
- Batch processing
- Results analysis

Run it:

```bash
cd llm_judge_package
python examples/basic_usage.py
```

## 🔧 Advanced Configuration

### Rate Limiting and Error Handling

All API clients include:

- Automatic retry with exponential backoff
- Rate limiting (1.5 second minimum between requests)
- Error recovery and reporting
- Cost tracking (for supported providers)

### Parallel Processing

```python
from processors import BatchProcessor

# Configure parallel processing
processor = BatchProcessor(
    judge,
    max_workers=4,      # Number of parallel threads
    batch_size=10       # Texts per batch
)

# Monitor progress
def progress_callback(completed, total):
    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

results = processor.process_texts(
    texts_list,
    metadata_list,
    progress_callback=progress_callback
)
```

### Custom API Clients

To add support for other providers, extend `BaseAPIClient`:

```python
from api_clients import BaseAPIClient

class MyAPIClient(BaseAPIClient):
    def __init__(self, api_key):
        super().__init__()
        # Initialize your client

    def send_request(self, model_name, prompt, max_tokens=150, temperature=0, **kwargs):
        # Implement your API call
        # Return the generated text as string
        pass
```

## 📈 Cost Management

Monitor API usage:

```python
# Check costs and success rates
print(f"Total cost: ${client.get_total_cost():.4f}")
print(f"Success rate: {client.get_success_rate():.1f}%")
print(f"Total requests: {client.success_count + client.error_count}")
```

Cost tracking is automatic for providers that report token usage. Providers lacking usage details currently return `$0.00`.

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to add the `src` directory to your Python path:

   ```python
   import sys
   sys.path.append('llm_judge_package/src')
   ```

2. **API Connection Issues**:
   - Verify your API credentials
   - Check endpoint URLs
   - Ensure model/deployment names are correct

3. **Rate Limiting**:
   - The package includes automatic rate limiting
   - Reduce `max_workers` in batch processing if needed
   - Increase delays between requests

4. **JSON Parsing Errors**:
   - The judge tries to extract JSON from LLM responses
   - If extraction fails, raw responses are returned
   - Consider adjusting the prompt or model temperature

### Getting Help

- Check the example scripts in the `examples/` directory
- Verify your taxonomy JSON files are valid
- Test API connections using the `verify_connection()` method
- Monitor error rates and success rates

## 🤝 Contributing

This package was extracted from the SanctSim cooperation research project. To contribute:

1. Add new API client implementations in `api_clients.py`
2. Extend taxonomy functionality in `taxonomy.py`
3. Add processing utilities in `processors.py`
4. Create example scripts for new features
5. Add unit tests in the `tests/` directory

## 📄 License

This package is provided as-is for research and educational purposes. Please ensure you comply with the terms of service of your chosen API provider.

## 🙏 Acknowledgments

Originally developed for analyzing LLM cooperation strategies in the SanctSim project. The cooperation taxonomy was designed to understand decision-making patterns in public goods games and sanctioning scenarios.
