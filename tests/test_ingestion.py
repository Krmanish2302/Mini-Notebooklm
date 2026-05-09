
---

## 6. ANTI-HALLUCINATION CHECKLIST

For implementation agents, follow this strictly:

### 6.1 Before Implementing
- [ ] Read the EXACT file path from this PRD
- [ ] Check if dependent files exist (import them)
- [ ] Verify function signatures match this PRD

### 6.2 During Implementation
- [ ] Write code for ONE file at a time
- [ ] Add type hints to all functions
- [ ] Add docstrings to all classes/methods
- [ ] Handle exceptions with try/except
- [ ] Never assume dependencies exist - import and check

### 6.3 After Implementing
- [ ] Create test notebook in `check/` folder
- [ ] Run the notebook - ALL cells must pass
- [ ] If any cell fails, fix before moving on
- [ ] Verify the file connects to pipeline (can be imported by master_pipeline.py)

### 6.4 Forbidden Actions
- [ ] NEVER skip testing after implementing a file
- [ ] NEVER change function signatures without updating all callers
- [ ] NEVER use global variables for state
- [ ] NEVER ignore exceptions with bare `except:`
- [ ] NEVER create files not in this PRD
- [ ] NEVER skip the `CONNECT TO PIPELINE` step

---

## 7. DEPENDENCIES

**File**: `requirements.txt`
