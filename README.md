# DeepReason AI Agent

An autonomous research engine powered by LangGraph, LangChain, and Tavily that transforms user queries into comprehensive research reports.

## Features

- **Autonomous Research Planning**: Breaks down complex queries into targeted search tasks
- **Multi-Source Research**: Executes searches using Tavily API for comprehensive data collection
- **Quality Control**: Built-in critic node evaluates research completeness and accuracy
- **Iterative Improvement**: Automatic revision loops to fill research gaps
- **Professional Reports**: AI-generated comprehensive reports with citations
- **Robust Error Handling**: Graceful fallbacks for API failures and edge cases

## Tech Stack

- **Language**: Python 3.9+
- **Orchestration**: LangGraph (StateGraph)
- **LLM**: LangChain (ChatOpenAI with gpt-4o-mini)
- **Search Tool**: Tavily API
- **Configuration**: python-dotenv

##  Project Structure

```
deep_reason/
├── .env                # API keys (OPENAI_API_KEY, TAVILY_API_KEY)
├── state.py            # AgentState TypedDict definition
├── nodes.py            # Four agent nodes: Planner, Researcher, Critic, Writer
├── main.py             # StateGraph workflow and conditional logic
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

##  Setup & Installation

1. **Clone and navigate to the project**:
   ```bash
   cd deep_reason
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**:
   Edit `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. **Run the agent**:
   ```bash
   python main.py
   ```

##  Workflow Architecture

The DeepReason agent follows a sophisticated 4-stage workflow:

### 1. **Planner Node**
- Analyzes user query
- Generates 3 distinct, targeted search queries
- Initializes research state

### 2. **Researcher Node**
- Executes search queries using Tavily API
- Accumulates search results with citations
- Handles API failures gracefully

### 3. **Critic Node** (Quality Control)
- Evaluates research completeness
- Determines if more information is needed
- Either proceeds to writing or generates new search queries
- Implements revision tracking to prevent infinite loops

### 4. **Writer Node**
- Synthesizes all research data
- Generates professional, well-structured reports
- Includes citations and references

## Conditional Logic Flow

```
planner → researcher → critic ──┐
             ↑                    │
             └─────(conditional)───┘
                                  ↓
                               writer → END
```

The conditional edge after the critic node implements intelligent decision-making:
- **Continue Research**: If critic rejects and revisions < max_revisions
- **Proceed to Writing**: If research is complete or max revisions reached

##  Usage Examples

### Basic Research Query
```
 Research Query: What are the latest developments in quantum computing?
```

### Complex Analysis
```
 Research Query: Compare the economic impacts of renewable energy policies in Europe vs North America
```

### Technical Investigation
```
 Research Query: How do transformer architectures work in natural language processing?
```

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for GPT-4o-mini access
- `TAVILY_API_KEY`: Required for web search functionality

### Agent Parameters
- `max_revisions`: Default 2 (prevents infinite loops)
- `temperature`: 0.7 (balanced creativity/reliability)
- `search_depth`: "basic" (Tavily search configuration)

## Error Handling

The agent includes comprehensive error handling:
- **API Failures**: Graceful fallbacks and retry logic
- **Invalid Queries**: Input validation and user guidance
- **Network Issues**: Timeout handling and error reporting
- **State Corruption**: Automatic state reset and recovery

##  Research Statistics

Each research session provides detailed statistics:
- Total revision cycles
- Number of content pieces collected
- Search queries executed
- Final report length and quality metrics

##  Advanced Usage

### Programmatic Integration
```python
from deep_reason.main import run_research

result = run_research("Your research query here")
print(result["final_report"])
```

### Custom Configuration
Modify the `AgentState` in `state.py` to add custom fields or adjust the workflow logic in `main.py`.

---

**Built with using LangGraph - The future of autonomous AI agents**
