import streamlit as st

discover_page = st.Page("discover.py", title="Discover.ai", icon="🔍")
define_page = st.Page("define.py", title="Define.ai", icon="📖") 
document_page = st.Page("etl_documentation.py", title="LineageExtractor.ai", icon="📝") 
model_page = st.Page("model.py", title="Model.ai", icon="🧬")
map_page = st.Page("map_rag_st.py", title="Map.ai", icon="↔️")
govern_page = st.Page("govern.py", title="Govern.ai", icon="✅")
test_page = st.Page("synthesize_sql.py", title="Test.ai", icon="🧪")
converse_page = st.Page("converse.py", title="Converse.ai", icon="💬")
storyteller_page = st.Page("storyteller.py", title="Storyteller.ai", icon="📊") 

pg = st.navigation({"Accelerators":[
    discover_page, 
    define_page, 
    document_page, 
    model_page,
    map_page, 
    govern_page, 
    test_page,
    converse_page,
    storyteller_page
    ]}
)

pg.run()

# st.title("AI for Data")
# st.subheader("Generative AI Accelerators", divider="violet")




