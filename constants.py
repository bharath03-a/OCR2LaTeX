DEFAULT_MESSAGE_CONTENT = """
Attached image contains one or multiple mathematical equations or structured tables. 
Your task is to accurately extract and convert them into LaTeX code. 

STRICT GUIDELINES (Failure to follow will result in penalties):
- **ONLY return valid LaTeX code.** Do not add explanations, descriptions, or modifications.
- **DO NOT simplify or alter the equations.** Preserve them exactly as they appear in the image.
- **DO NOT provide definitions or interpretations** of symbols.
- **Output must be in proper LaTeX syntax** with appropriate mathematical formatting for mathematical equations or tabular formatting for structured tables.
- **DO NOT include any additional content** in the output.
"""