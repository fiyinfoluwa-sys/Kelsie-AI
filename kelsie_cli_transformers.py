import torch
import json
import os
import datetime
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import urllib.parse

class KelsieAI:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        self.conversation_history = []
        
        self.base_context = """You are Kelsie, a knowledgeable AI assistant. You provide accurate, factual information and have coherent conversations. You always use proper grammar.

Current date: {date}
Current time: {time}

Recent conversation:
{history}

User: {input}
Kelsie:"""

    def google_search(self, query):
        """Perform a Google search using the Custom Search JSON API"""
        try:
            # Using your provided Google API credentials
            api_key = "AIzaSyBgN_yDe8p2b8sEBi-Q6p1uQuFd5w43mT0"
            search_engine_id = "947e2fc0ae3c9450c"
            
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': api_key,
                'cx': search_engine_id,
                'q': query,
                'num': 3
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'items' in data and len(data['items']) > 0:
                top_result = data['items'][0]
                snippet = top_result.get('snippet', '')
                title = top_result.get('title', '')
                return f"{title}. {snippet}"
                
        except Exception as e:
            print(f"Google search error: {e}")
            
        return self.fallback_search(query)

    def fallback_search(self, query):
        """Fallback search using DuckDuckGo"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            
            if data.get('AbstractText'):
                return data['AbstractText']
            elif data.get('Answer'):
                return data['Answer']
            elif data.get('RelatedTopics'):
                return data['RelatedTopics'][0]['Text'] if data['RelatedTopics'] else None
        except:
            pass
        return None

    def get_web_knowledge(self, user_input):
        """Get factual information from the web"""
        user_lower = user_input.lower()
        
        # Direct answers for common questions
        if any(word in user_lower for word in ['how are you', 'how do you feel']):
            return "I'm functioning well, thank you. How can I help you today?"
        
        elif any(word in user_lower for word in ['date', 'today\'s date']):
            return f"Today's date is {datetime.datetime.now().strftime('%B %d, %Y')}."
        
        elif any(word in user_lower for word in ['time', 'current time']):
            return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}."
        
        elif any(word in user_lower for word in ['who are you', 'what are you']):
            return "I'm Kelsie, an AI assistant that can access current information from the web to answer your questions accurately."
        
        # Search the web for factual information
        factual_keywords = ['who is', 'what is', 'when was', 'where is', 'president of', 'prime minister', 'capital of', 'population of', 'weather in', 'news about']
        if any(phrase in user_lower for phrase in factual_keywords):
            web_result = self.google_search(user_input)
            if web_result:
                return f"Based on current information: {web_result}"
            else:
                return "I couldn't find current information on that topic. Could you be more specific?"
        
        return None

    def build_context(self, user_input):
        now = datetime.datetime.now()
        
        history_text = ""
        for i, (user, bot) in enumerate(self.conversation_history[-4:]):
            history_text += f"User: {user}\nKelsie: {bot}\n"
        
        context = self.base_context.format(
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H:%M"),
            history=history_text,
            input=user_input
        )
        
        return context

    def format_response(self, text):
        """Ensure proper capitalization and punctuation"""
        if not text:
            return "Could you clarify that for me?"
        
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        if text and not text.endswith(('.', '!', '?')):
            if any(word in text.lower() for word in ['who', 'what', 'when', 'where', 'why', 'how']):
                text += '?'
            else:
                text += '.'
        
        return text

    def generate_response(self, user_input):
        # First check for web-based knowledge
        web_response = self.get_web_knowledge(user_input)
        if web_response:
            response = self.format_response(web_response)
            self.conversation_history.append((user_input, response))
            return response
        
        # Build context for the model
        context = self.build_context(user_input)
        
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        if input_ids.shape[1] > 800:
            input_ids = input_ids[:, -800:]
        
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            response_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=input_ids.shape[1] + 80,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.85,
                top_k=30,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        response = self.tokenizer.decode(
            response_ids[:, input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        ).strip()
        
        response = self.clean_response(response)
        response = self.format_response(response)
        
        self.conversation_history.append((user_input, response))
        if len(self.conversation_history) > 6:
            self.conversation_history.pop(0)
        
        return response

    def clean_response(self, response):
        if not response:
            return "I'm not sure I understand. Could you rephrase that?"
        
        # Remove any reference to the user speaking
        stop_phrases = ['User:', 'user:', 'Human:', 'human:', 'Kelsie:', 'kelsie:']
        for phrase in stop_phrases:
            if phrase in response:
                response = response.split(phrase)[0].strip()
        
        # Remove inappropriate content
        inappropriate = ['love u', 'i love you', 'you\'re hot', 'sexy', 'hate you']
        if any(phrase in response.lower() for phrase in inappropriate):
            return "I'm here to provide helpful information. How can I assist you?"
        
        if len(response) < 2:
            return "Could you tell me more about that?"
        
        return response

def main():
    print("Kelsie AI Initializing...")
    kelsie = KelsieAI()
    print("Kelsie AI Ready - Web-Connected AI Assistant")
    print("Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Kelsie: Goodbye! Feel free to ask me anything anytime.")
            break
        
        if not user_input:
            continue
            
        response = kelsie.generate_response(user_input)
        print(f"Kelsie: {response}")

if __name__ == "__main__":
    main()
