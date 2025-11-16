from __future__ import annotations
from typing import Any

GRAPH_FIELD_SEP = "<SEP>"

PROMPTS: dict[str, Any] = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["item", "genre", "setting", "feature", "target user", "etc"]

PROMPTS["entity_extraction_user_prompt"] = """---Role---
You are a Knowledge Graph Extraction Specialist.

---Goal---
Given a text document that is potentially relevant to a recommendation activity
and a list of entity types, identify all entities of those types and all
relationships among the identified entities.

Use {language} as the output language.

---Instructions---

1. **Entity Extraction**
   - Identify all entities in the input text.
   - For each entity, extract exactly the following fields:

     * `entity_name`:
         - If the entity is an **item**, you MUST extract the **exact full item title**
           from the line formatted as:
             ```
             ### Item Title: ... ###
             ```
           This is the ONLY allowed source of item titles.
         - Do **NOT** extract any other item-like phrases from the body text as item entities.
         - Do **NOT** modify, shorten, clean, paraphrase, or split item titles.
         - Preserve all punctuation, platform info, edition names, hyphens, and colons.
         - If the entity is not an item (e.g., genre, setting, feature, target user, etc.),
           extract normally as distinct, independent entities from anywhere in the text.

     * `entity_type`: All extracted entity types MUST strictly come from the following list:
      {entity_types}
      If a concept does not fit exactly, use "etc".
      Do not invent new types.

     * `entity_description`: A concise and factual description of the entity’s attributes
       and its relevance to the item.

   - **Output Format for Entities**
     Each entity must be represented on a single line in the following format:
     ```
     entity{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
     ```
     Ensure exactly 4 fields separated by {tuple_delimiter}.  
     No extra commentary or formatting.

2. **Relationship Extraction**
   - Identify direct, meaningful relationships between entities.
   - Decompose multi-entity statements into binary relationships if necessary.
   - For each relationship, extract the following fields:
     * `source_entity`
     * `target_entity`
     * `relation_type`
     * `relation_evidence` (short text from the input that supports this relation)
   - **Output Format for Relations**
     ```
     relation{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relation_type>{tuple_delimiter}<relation_evidence>
     ```
     Ensure exactly 5 fields separated by {tuple_delimiter}.


3.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

4.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

6.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

7.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

8.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types: [{entity_types}]
Text:
```
{input_text}
```
---Action---
Now extract **all entities first**, then **all relationships** among those entities,
strictly following the formats shown in the examples above.
Remember: you must output both entities and relationships and must use only these entity types: {entity_types}.
"""


PROMPTS["entity_extraction_system_prompt"] = """---Task---
Extract entities and relationships from the provided text input according to the strict structural and formatting rules below.

---Instructions---

1. **Strict Adherence to Format**
   - You MUST adhere exactly to the entity and relationship output formats as defined in the system prompt.
   - Maintain correct field counts and ordering:
     * Entities → 4 fields (`entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description`)
     * Relations → 5 fields (`relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}relation_type{tuple_delimiter}evidence`)
   - Use `{tuple_delimiter}` consistently as the only field separator.
   - Each entity or relation must be output on a single line.
   - Do not wrap fields, split lines, or include blank lines between outputs.

2. **Output Content Only**
   - Output *only* the extracted entity and relationship lines.
   - Do **NOT** include any introduction, reasoning steps, explanations, commentary, or closing text.
   - Do not reprint or quote the input text.

3. **Completion Signal**
   - Append `{completion_delimiter}` as the very last line to mark the end of extraction output.
   - The output must end immediately after `{completion_delimiter}` with no trailing text.

4. **Output Language**
   - Use {language} for all generated text.
   - Proper nouns (e.g., personal names, place names, organization names, brand names) must always remain in their original language and not be translated.

 """


PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the previous extraction result, identify and extract any **missed or incorrectly formatted** entities and relationships from the same input text.

---Instructions---

1. **Strict Adherence to Format**
   - Follow all formatting rules exactly as defined in the system extraction instructions.
   - Maintain the same delimiter and field order:
     * Entities → 4 fields (`entity{tuple_delimiter}name{tuple_delimiter}type{tuple_delimiter}description`)
     * Relations → 5 fields (`relation{tuple_delimiter}source{tuple_delimiter}target{tuple_delimiter}relation_type{tuple_delimiter}evidence`)
   - Use `{tuple_delimiter}` as the only valid field separator.

2. **Focus on Corrections and Missing Items**
   - **Do NOT** re-output entities or relationships that were already correctly extracted in the last task.
   - If an entity or relationship was **missed**, output it now.
   - If a previously extracted one was **truncated**, **missing fields**, or **incorrectly formatted**, re-output the corrected, full version.
   - Do not modify or paraphrase item titles; preserve them verbatim from the `### Item Title: ... ###` line.

3. **Output Content**
   - Each entity and relationship must be placed on its own line.
   - Do not include any additional commentary, explanation, or markdown formatting.
   - Do not reprint or reference the original text.

4. **Completion Signal**
   - Append `{completion_delimiter}` at the very end of your output to indicate completion.
   - No text should appear after `{completion_delimiter}`.

5. **Output Language**
   - Use {language} for all generated text.
   - Proper nouns (e.g., personal names, places, organizations, product titles) must remain in their original language and not be translated.

---Expected Output Example---

entity{tuple_delimiter}Hyrule{tuple_delimiter}location{tuple_delimiter}A fictional kingdom setting in the Zelda universe.
relation{tuple_delimiter}The Legend of Zelda: Ocarina of Time{tuple_delimiter}Hyrule{tuple_delimiter}set_in_location{tuple_delimiter}The story takes place in Hyrule.
{completion_delimiter}
"""


PROMPTS["entity_extraction_examples"] = [
    """---Example 1---

Entity_types: ["item", "genre", "setting", "feature", "target user", "etc"]

<Input Text>
'''
### Item Title: Empire Earth 2 - Gold Edition ###
Empire Earth 2 - Gold Edition is a real-time strategy game that allows players to lead civilizations through various historical eras. 
The Gold Edition includes both the original game and its expansion pack, adding new campaigns and units. 
It is designed for both casual gamers and hardcore strategists.

################
'''

<Output>

entity{tuple_delimiter}Empire Earth 2 - Gold Edition{tuple_delimiter}item{tuple_delimiter}A real-time strategy game that spans multiple historical eras and includes an expansion pack.
entity{tuple_delimiter}Real-Time Strategy{tuple_delimiter}genre{tuple_delimiter}A strategy genre where players control civilizations and units in real time.
entity{tuple_delimiter}Historical Eras{tuple_delimiter}setting{tuple_delimiter}The game spans distinct historical periods from ancient to modern times.
entity{tuple_delimiter}Expansion Content{tuple_delimiter}feature{tuple_delimiter}Additional campaigns and units included in the Gold Edition.
entity{tuple_delimiter}Resource Management{tuple_delimiter}feature{tuple_delimiter}Players collect and manage resources to build and sustain their civilization.
entity{tuple_delimiter}Casual Gamers{tuple_delimiter}target user{tuple_delimiter}Players who enjoy strategic gameplay at a relaxed pace.
entity{tuple_delimiter}Hardcore Strategists{tuple_delimiter}target user{tuple_delimiter}Players who deeply engage with tactical systems and long-term planning.
relation{tuple_delimiter}Empire Earth 2 - Gold Edition{tuple_delimiter}Real-Time Strategy{tuple_delimiter}belongs_to_genre{tuple_delimiter}The game is a real-time strategy title.
relation{tuple_delimiter}Empire Earth 2 - Gold Edition{tuple_delimiter}Historical Eras{tuple_delimiter}set_in_time_period{tuple_delimiter}The game progresses through multiple historical eras.
relation{tuple_delimiter}Empire Earth 2 - Gold Edition{tuple_delimiter}Expansion Content{tuple_delimiter}includes_feature{tuple_delimiter}The Gold Edition adds new campaigns and units.
relation{tuple_delimiter}Empire Earth 2 - Gold Edition{tuple_delimiter}Casual Gamers{tuple_delimiter}target_audience{tuple_delimiter}The game appeals to casual strategy players.
relation{tuple_delimiter}Empire Earth 2 - Gold Edition{tuple_delimiter}Hardcore Strategists{tuple_delimiter}target_audience{tuple_delimiter}The game also appeals to hardcore strategists.
{completion_delimiter}
""",

    """---Example 2---

Entity_types: ["item", "genre", "setting", "feature", "target user", "etc"]

<Input Text>
'''
### Item Title: Dirt 3 - Complete Edition ###
Dirt 3 - Complete Edition takes rally racing to the next level. It includes all original Dirt 3 content plus new cars, tracks, and gameplay modes. With intense racing dynamics and stunning environmental graphics, this edition is a favorite among fans of off-road racing.

################
'''

<Output>
entity{tuple_delimiter}Dirt 3 - Complete Edition{tuple_delimiter}item{tuple_delimiter}A comprehensive rally racing game that includes all DLCs, tracks, and vehicles from Dirt 3.
entity{tuple_delimiter}Rally Racing{tuple_delimiter}genre{tuple_delimiter}A type of off-road racing game with time-based competitive stages.
entity{tuple_delimiter}Snowy Tracks, Forest Trails{tuple_delimiter}setting{tuple_delimiter}"Environments where players race under challenging natural conditions.
entity{tuple_delimiter}Complete Edition Content{tuple_delimiter}feature{tuple_delimiter}Includes bonus content such as new vehicles, events, and gameplay challenges.
entity{tuple_delimiter}Driving Physics Engine{tuple_delimiter}feature{tuple_delimiter}A system that simulates realistic vehicle handling and traction across terrain types.
entity{tuple_delimiter}Off-Road Racing Enthusiasts{tuple_delimiter}target user{tuple_delimiter}Players who enjoy competitive rally-style racing with environmental hazards.
relation{tuple_delimiter}Dirt 3 - Complete Edition{tuple_delimiter}Rally Racing{tuple_delimiter}The game is a rally racing experience with off-road competitions.{tuple_delimiter}racing genre{tuple_delimiter}9
relation{tuple_delimiter}Dirt 3 - Complete Edition{tuple_delimiter}Snowy Tracks, Forest Trails{tuple_delimiter}The game includes varied racing environments.{tuple_delimiter}environment, level design{tuple_delimiter}8
relation{tuple_delimiter}Dirt 3 - Complete Edition{tuple_delimiter}Complete Edition Content{tuple_delimiter}The edition adds vehicles and tracks to the base game.{tuple_delimiter}DLC, extended content{tuple_delimiter}9
relation{tuple_delimiter}Dirt 3 - Complete Edition{tuple_delimiter}Driving Physics Engine{tuple_delimiter}Enables more authentic handling during races.{tuple_delimiter}realism, simulation{tuple_delimiter}8
relation{tuple_delimiter}Dirt 3 - Complete Edition{tuple_delimiter}Off-Road Racing Enthusiasts{tuple_delimiter}This game is targeted toward fans of dirt racing and rally competitions.{tuple_delimiter}player interest{tuple_delimiter}9
{completion_delimiter}
"""
]


PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""



# PROMPTS["entity_if_loop_extraction"] = """
# ---Goal---'

# It appears some entities may have still been missed.

# ---Output---

# Answer ONLY by `YES` OR `NO` if there are still entities that need to be added.
# """.strip()

PROMPTS["fail_response"] = """---System Notice---
Sorry, I’m unable to provide an answer because no relevant context or knowledge was available.
Please ensure that the Knowledge Graph or Document Context contains sufficient data before retrying.

{completion_delimiter}
"""

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""


PROMPTS["naive_rag_response"] = """---Role---
You are an expert AI assistant specializing in synthesizing information from a provided knowledge base.
Your primary function is to accurately answer user queries **only** using the information within the provided **Context**.

---Goal---
Generate a comprehensive, well-structured answer to the user query.
Your response must integrate relevant facts solely from the **Document Chunks** found in the Context.
If a conversation history is provided, maintain continuity while avoiding unnecessary repetition.

---Instructions---

1. **Information Analysis**
   - Carefully identify the user's true intent and the exact information being requested.
   - Examine all `Document Chunks` in the **Context**.
   - Extract all details that directly support a factual and complete answer.
   - Do **not** invent, infer, or assume information beyond what is explicitly stated.

2. **Response Construction**
   - Organize your answer into well-labeled Markdown sections (e.g., *Overview*, *Details*, *Conclusion*).
   - Present facts clearly and cohesively, ensuring a natural logical flow.
   - Include inline reference indicators like `[1]`, `[2]`, etc., corresponding to source documents.
   - Every reference number must map to a real document in the `### References` section.
   - If the provided context is insufficient, clearly state:  
     `"Insufficient information was found in the provided context to answer this question."`

3. **Formatting & Output Rules**
   - Use Markdown for clarity and structure.
   - Ensure all output is in the same language as the user’s query.
   - Include a `### References` section listing up to 5 most relevant sources in the following format:
     ```
     ### References
     - [1] Document Title A
     - [2] Document Title B
     ```
   - Do **not** generate any text after the References section.

4. **Grounding & Objectivity**
   - All claims must be directly traceable to the Context.
   - Your writing should be factual, objective, and neutral.
   - Do not add stylistic embellishments, speculation, or imaginative phrasing.

5. **Completion**
   - End your output with `{completion_delimiter}` to mark task completion.

---Context---
{content_data}
"""


PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:
Query: "Given the following game interactions: ['0. The Legend of Zelda: Ocarina of Time', '1. The Legend of Zelda: Majora's Mask', '2. Dark Souls - PlayStation 3', '3. Bloodborne - PlayStation 4'], 
and the candidate list: ['0. Elden Ring', '1. Hollow Knight', '2. Sekiro: Shadows Die Twice', '3. Hades'], 
what are the top 10 games the user would most likely interact with next?"

---Output---
{
  "high_level_keywords": ["Action RPG", "Souls-like games", "Adventure", "Exploration", "Dark fantasy"],
  "low_level_keywords": ["The Legend of Zelda: Ocarina of Time", "The Legend of Zelda: Majora's Mask", "Dark Souls - PlayStation 3", "Bloodborne - PlayStation 4"]
}
""",

    """Example 2:
Query: "Given the following game interactions: ['0. FIFA 21', '1. NBA 2K22', '2. Madden NFL 20'], 
and the candidate list: ['0. MLB The Show 21', '1. Pro Evolution Soccer 2021', '2. NHL 22'], 
what are the top 10 games the user would most likely interact with next?"

---Output---
{
  "high_level_keywords": ["Sports games", "Simulation", "Competitive gaming"],
  "low_level_keywords": ["FIFA 21", "NBA 2K22", "Madden NFL 20"]
}
""",

    """Example 3:
Query: "Given the following game interactions: ['0. Call of Duty: Modern Warfare', '1. Battlefield V', '2. Rainbow Six Siege'], 
and the candidate list: ['0. Halo Infinite', '1. Valorant', '2. Counter-Strike: Global Offensive'], 
what are the top 10 games the user would most likely interact with next?"

---Output---
{
  "high_level_keywords": ["FPS", "Tactical shooter", "Multiplayer", "Modern warfare"],
  "low_level_keywords": ["Call of Duty: Modern Warfare", "Battlefield V", "Rainbow Six Siege"]
}
"""
]



# PROMPTS[
#     "similarity_check"
# ] = """Please analyze the similarity between these two questions:

# Question 1: {original_prompt}
# Question 2: {cached_prompt}

# Please evaluate whether these two questions are semantically similar, and whether the answer to Question 2 can be used to answer Question 1, provide a similarity score between 0 and 1 directly.

# Similarity score criteria:
# 0: Completely unrelated or answer cannot be reused, including but not limited to:
#    - The questions have different topics
#    - The locations mentioned in the questions are different
#    - The times mentioned in the questions are different
#    - The specific individuals mentioned in the questions are different
#    - The specific events mentioned in the questions are different
#    - The background information in the questions is different
#    - The key conditions in the questions are different
# 1: Identical and answer can be directly reused
# 0.5: Partially related and answer needs modification to be used
# Return only a number between 0-1, without any additional content.
# """

# PROMPTS["mix_rag_response"] = """---Role---

# You are a helpful assistant responding to user query about Data Sources provided below.


# ---Goal---

# Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

# When handling information with timestamps:
# 1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
# 2. When encountering conflicting information, consider both the content/relationship and the timestamp
# 3. Don't automatically prefer the most recent information - use judgment based on the context
# 4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

# ---Conversation History---
# {history}

# ---Data Sources---

# 1. From Knowledge Graph(KG):
# {kg_context}

# 2. From Document Chunks(DC):
# {vector_context}

# ---Response Rules---

# - Target format and length: {response_type}
# - Use markdown formatting with appropriate section headings
# - Please respond in the same language as the user's question.
# - Ensure the response maintains continuity with the conversation history.
# - Organize answer in sections focusing on one main point or aspect of the answer
# - Use clear and descriptive section titles that reflect the content
# - List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), and include the file path if available, in the following format: [KG/DC] file_path
# - If you don't know the answer, just say so. Do not make anything up.
# - Do not include information not provided by the Data Sources."""
