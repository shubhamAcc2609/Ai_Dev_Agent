# AI Dev Agent

An autonomous software development agent built using LangGraph that transforms natural language requirements into executable software projects through intelligent planning, routing, execution, and validation.

---

## Overview

AI Dev Agent explores the concept of autonomous software engineering by combining Large Language Models with workflow orchestration. The system accepts a user requirement, generates a structured implementation plan, determines the most appropriate execution strategy, builds the project, validates the output, and provides execution feedback.

The architecture is designed to move beyond simple code generation and demonstrate how AI agents can participate in a larger portion of the software development lifecycle.

---

## Architecture

```text
                User Requirement
                        │
                        ▼
               ┌─────────────────┐
               │  Planner Node   │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   Router Node   │
               └────────┬────────┘
                        │
      ┌─────────────────┼─────────────────┐
      │                 │                 │
      ▼                 ▼                 ▼
┌────────────┐   ┌────────────┐   ┌────────────┐
│  Simple    │   │ Compiled   │   │    Web     │
│ Executor   │   │ Executor   │   │ Executor   │
└─────┬──────┘   └─────┬──────┘   └─────┬──────┘
      │                │                │
      └────────────────┴────────────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   Critic Node   │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │ Toolchain Check │
               └─────────────────┘
```

---

## Features

### Intelligent Planning

- Converts natural language requirements into structured implementation plans.
- Identifies:
  - Programming language
  - Application type
  - Dependencies
  - Build requirements
  - Runtime requirements

### Dynamic Routing

The Router Node analyzes planner output and selects the correct execution pathway.

Supported categories:

- Python scripts
- Console applications
- FastAPI applications
- Flask applications
- C projects
- C++ projects
- Rust projects

### Specialized Executors

#### Simple Executor

Handles lightweight projects such as:

- Python scripts
- Utility programs
- Command-line applications

Features:

- Safe command execution
- Timeout protection
- Process monitoring
- Artifact collection

#### Compiled Executor

Handles compiled languages.

Supported:

- C
- C++
- Rust

Features:

- Compiler detection
- Build automation
- Compilation diagnostics
- Executable validation

#### Web Executor

Handles server-based applications.

Supported:

- FastAPI
- Flask

Features:

- Dependency installation
- Automatic server launch
- Health checks
- Endpoint testing
- OpenAPI validation

### Critic Node

The Critic Node introduces reflection into the workflow.

Responsibilities:

- Analyze execution results
- Detect failures
- Review generated artifacts
- Suggest improvements
- Improve system reliability

### Toolchain Detection

Verifies required development tools before execution.

Checks may include:

- Python
- Pip
- GCC/G++
- Rust
- Virtual environments
- Framework dependencies

---

## Workflow

### 1. User Requirement

Example:

```text
Create a FastAPI weather API that returns current weather for a city.
```

### 2. Planning

The Planner Node creates:

- Project structure
- Implementation steps
- Required files
- Dependency list

### 3. Routing

The Router Node determines:

```text
Project Type: Web Service
Framework: FastAPI

Route → Web Executor
```

### 4. Execution

Selected executor:

- Generates code
- Creates project files
- Installs dependencies
- Runs application

### 5. Validation

The system verifies:

- Build success
- Runtime success
- Endpoint accessibility
- Output correctness

### 6. Critique

The Critic Node evaluates results and suggests improvements.

---

## Project Structure

```text
Ai_Dev_Agent/
│
├── src/
│   └── agent/
│       ├── graph.py
│       ├── state.py
│       │
│       └── nodes/
│           ├── planner.py
│           ├── router.py
│           ├── simple_executor.py
│           ├── compiled_executor.py
│           ├── web_executor.py
│           ├── critic.py
│           └── toolchain_detector.py
│
├── generated_projects/
├── tests/
├── requirements.txt
├── README.md
└── main.py
```

---

## Technology Stack

- Python 3.11+
- LangGraph
- LangChain
- OpenAI / Azure OpenAI Compatible Models
- FastAPI
- Flask
- GCC / G++
- Rust Toolchain

---

## Example Output

```text
Status: SUCCESS

Plan Steps: 4

Files Created:
- main.py
- requirements.txt

Server Started Successfully

Validation Results:
✓ Dependency Installation
✓ Server Startup
✓ API Endpoint Test
✓ OpenAPI Check
```

---

## Current Roadmap

### Completed

- Planner Node
- Router Node
- Simple Executor
- Compiled Executor
- Web Executor
- Dynamic Execution Routing

### In Progress

- Critic Node
- Toolchain Detector

### Future Enhancements

- Conditional workflow branching
- Retry mechanisms
- Multi-agent collaboration
- Memory systems
- Advanced orchestration patterns

---

## Learning Objectives

This project explores:

- Autonomous software development
- Workflow orchestration
- AI planning systems
- Dynamic routing architectures
- Execution-aware agents
- Reflection-based feedback loops

---

## Author

**Shubham**

Custom Software Engineering Associate

GitHub: https://github.com/shubhamAcc2609

---

## License

This project is intended for educational, research, and experimentation purposes.