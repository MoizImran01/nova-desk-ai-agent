from typing import List
from langchain_core.documents import Document
import difflib
def format_faqs(docs: List[Document])->str:
    formatted_faqs = []
    for doc in docs:
        page_number = doc.metadata.get("page", "N/A")
        content = doc.page_content
        title = doc.metadata.get("source", "N/A")
        formatted_faqs.append(f"Page Number: {page_number}\n Title: {title}\nContent: {content}")
    return "\n\n---\n\n".join(formatted_faqs)

def return_closest_match(user_request_service_name: str, valid_service_names: List[str]) -> List[str]:
    closest_matches = difflib.get_close_matches(
            user_request_service_name.lower(), 
            [name.lower() for name in valid_service_names], 
            n=1, 
            cutoff=0.5
        )

    return closest_matches