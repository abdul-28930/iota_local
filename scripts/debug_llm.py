import sys
sys.path.insert(0, ".")
import llm_local

resp = llm_local.raw_generate("Return only this JSON, nothing else: {\"test\": \"hello\"}", max_tokens=100)
print("RAW RESPONSE:", repr(resp))
