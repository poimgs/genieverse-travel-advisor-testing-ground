import os
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatManager:
    def __init__(self):
        """Initialize the ChatManager with an empty conversation history."""
        self.conversation_history = []
        self.token_usage = {
            'total_prompt_tokens': 0,
            'total_completion_tokens': 0,
            'total_tokens': 0,
            'total_cost': 0.0
        }
        # Default editable system prompt
        self.editable_system_prompt = """You are a helpful and knowledgeable travel advisor specializing in Ang Mo Kio, Bedok, Joo Chiat, Katong, Orchard, Tiong Bahru, Walking Trails in Singapore. You will engage travelers in **friendly, in-depth conversations** to understand their needs before offering guidance. Your goal is to provide **personalized, accurate, and engaging** advice about traveling in Singapore, while focusing on **asking questions** and **building rapport** rather than immediately listing suggestions.

### IMPORTANT GUIDELINES:

1. **Use Full and Exact Location Names**  
   - Always mention the **full** name of any location **exactly** as provided in the LOCATION field. **Double-check** the names before mentioning them to ensure accuracy.  
   - Do not abbreviate, alter, or modify location names.

2. **Strict Context Adherence**  
   - **Do not** suggest or provide any locations that are not explicitly mentioned in the RELEVANT LOCATION INFORMATION.  
   - If a traveler asks about a location not covered in the context, **do not provide any suggestions** related to that location. Politely acknowledge the limitation: "I currently don't have information on that location, but I can offer suggestions from the list of places in Singapore provided."

3. **In-depth Exploration Before Suggesting**  
   - Ask follow-up questions (e.g., about interests, duration of stay, preferred activities) **before** offering suggestions. This ensures your advice is tailored and stays within the provided context.  
   - Focus on the traveler's goals, previous experiences, and any special preferences.

4. **Location-Context Check**  
   - Always cross-check every suggestion you make to ensure it is part of the **RELEVANT LOCATION INFORMATION** section. If a location is not included in the context, immediately **disqualify it from being suggested**.
   - If you're uncertain about whether a location belongs to the context, **ask for clarification** from the traveler.

5. **Structured Recommendations (Max 3)**  
   - Provide suggestions only after gathering enough details to give truly tailored advice.  
   - If you recommend multiple places, present them as a **clear, numbered list** (1–3 items total).  
   - Briefly explain **why** each recommendation fits the traveler's interests.

6. **Concise & Conversational Tone**  
   - Be warm, approachable, and respectful, like a friendly local guide.  
   - Focus on the most relevant details without overwhelming the traveler.

7. **Accurate & Context-Only Responses**  
   - Only respond with information contained in the RELEVANT LOCATION INFORMATION section.  
   - If the traveler asks for something outside the provided context, politely acknowledge the knowledge limitation and provide alternatives from the context.

8. **No Fabrication**  
   - Do **not** invent or alter details about any locations beyond what is provided in the context.

9. **Genuine Engagement**  
   - Show curiosity by asking clarifying questions whenever helpful (e.g., "Could you tell me more about what you enjoy?").  
   - Only provide recommendations after you understand the traveler's needs well.

10. **Respect Specific Requests**  
   - If the user specifically requests **local options**, avoid suggesting other international or non-local cuisines unless the user explicitly indicates an interest in them.  
   - Always align your advice with the traveler's stated preferences and clarify if you are unsure.

### TONE & STYLE:  
- Be **warm, welcoming, and conversational**—like chatting with a friendly local.  
- Adapt the depth and style of your responses based on whether the user is a first-time visitor or a frequent traveler.  
- Maintain a relaxed, open-ended approach that fosters conversation and encourages the traveler to share more details about their trip.  
- **Never exceed three recommendations** in any response."""
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ('user', 'assistant', or 'system')
            content: The content of the message
        """
        self.conversation_history.append({"role": role, "content": content})
    
    # TODO: Update the detils, we may want to just include all details...
    def format_location_for_context(self, location: Dict[str, Any]) -> str:
        """
        Format a location dictionary into a string for context.
        
        Args:
            location: Dictionary containing location information
            
        Returns:
            Formatted string
        """
        return f"""
LOCATION: {location['title']}
AREA: {location['Location / Area']}
CATEGORY: {location['Category / Type']}
THEMES: {location['Theme / Highlights']}
PRICE: {location['Price Range']}
AUDIENCE: {location['Audience / Suitability']}
HOURS: {location['Operating Hours']}
ATTRIBUTES: {location['Additional Attributes']}
DETAILS: {location['content']}
"""
    
    def get_editable_system_prompt(self) -> str:
        """
        Get the current editable portion of the system prompt.
        
        Returns:
            The editable system prompt string
        """
        return self.editable_system_prompt
    
    def set_editable_system_prompt(self, new_prompt: str) -> None:
        """
        Update the editable portion of the system prompt.
        
        Args:
            new_prompt: The new editable system prompt
        """
        self.editable_system_prompt = new_prompt
    
    def generate_system_prompt(self, locations: List[Dict[str, Any]]) -> str:
        """
        Generate a system prompt with context from similar locations.
        
        Args:
            locations: List of similar locations
            
        Returns:
            System prompt string
        """
        # Format the location information
        context = "\n\n".join([self.format_location_for_context(loc) for loc in locations])
        
        # The location information section that must be preserved
        location_info_section = f"""### RELEVANT LOCATION INFORMATION:  
{context}  
**ONLY recommend locations mentioned above in the RELEVANT LOCATION INFORMATION section.**
"""
        
        # Combine the editable prompt with the fixed location information
        system_prompt = f"{self.editable_system_prompt}\n\n{location_info_section}"
        
        return system_prompt
    
    def generate_response(self, query: str, similar_locations: List[Dict[str, Any]]) -> str:
        """
        Generate a response to the user's query using OpenAI.
        
        Args:
            query: The user's query
            similar_locations: List of similar locations
            
        Returns:
            Generated response
        """
        
        # Generate system prompt with context
        system_prompt = self.generate_system_prompt(similar_locations)
        
        # Add the user's query to the conversation history
        self.add_message("user", query)
        
        # Prepare messages for the API call
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add the last few messages from the conversation history (to keep context manageable)
        # max_history = 5  # Adjust as needed
        # for message in self.conversation_history[-max_history:]:
        #     messages.append(message)

        for message in self.conversation_history:
            messages.append(message)
        
        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # or another appropriate model TODO: Move this to env variable to change and test
                messages=messages,
                temperature=0.7,
                # max_tokens=800
            )
            
            # Extract the response content
            response_content = response.choices[0].message.content
            
            # Track token usage
            usage = response.usage
            self.token_usage['total_prompt_tokens'] += usage.prompt_tokens
            self.token_usage['total_completion_tokens'] += usage.completion_tokens
            self.token_usage['total_tokens'] += usage.total_tokens
            # Approximate cost calculation (adjust rates as needed)
            prompt_cost = usage.prompt_tokens * 0.00000015  # $0.150 / 1M tokens
            completion_cost = usage.completion_tokens * 0.0000006  # $0.600 / 1M tokens
            self.token_usage['total_cost'] += prompt_cost + completion_cost
            
            # Add the assistant's response to the conversation history
            self.add_message("assistant", response_content)
            
            return response_content
        
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            print(error_message)
            return error_message
    
    def reset_conversation(self) -> None:
        """Reset the conversation history and user preferences."""
        self.conversation_history = []