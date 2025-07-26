import streamlit as st
import base64
import json
import os
from typing import List, Set, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
import pandas as pd
from PIL import Image
import io

# Pydantic Models for structured data extraction
class ClothingItem(BaseModel):
    category: str = Field(..., description="Category of the clothing item (e.g., T-Shirt, Dress, Pants, Shorts)")
    description: str = Field(..., description="Detailed description of the clothing item")
    color: List[str] = Field(..., description="Available colors of the clothing item")
    gender: str = Field(..., description="Gender suitability (Unisex, Male, Female)")
    fabric: str = Field(..., description="Type of fabric used")
    pattern: str = Field(..., description="Pattern (Solid, Striped, Checked, Floral, etc.)")
    fit: str = Field(..., description="Fit type (Regular Fit, Slim Fit, Loose Fit)")
    sleeve_length: str = Field(..., description="Sleeve length (Short, Long, 3/4, Sleeveless, N/A)")
    neck_type: str = Field(..., description="Neck type (Round, V-Neck, Collar, etc.)")
    occasion: List[str] = Field(..., description="Suitable occasions")
    season: List[str] = Field(..., description="Suitable seasons")
    features: Set[str] = Field(..., description="Special features")

class OutfitRecommendation(BaseModel):
    recommended_items: List[str] = Field(..., description="List of clothing item IDs for the recommended outfit")
    reasoning: str = Field(..., description="Explanation for why this outfit was recommended")
    style_tips: List[str] = Field(..., description="Additional styling tips")

class StyleSyncBot:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            api_key=api_key,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.structured_llm = self.llm.with_structured_output(ClothingItem, method="json-mode")
        self.recommendation_llm = self.llm.with_structured_output(OutfitRecommendation, method="json-mode")
        self.wardrobe = []

    def encode_image(self, image_bytes):
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode("utf-8")

    def analyze_clothing_image(self, image_bytes):
        """Analyze uploaded clothing image and extract structured data"""
        try:
            image_base64 = self.encode_image(image_bytes)
            
            prompt = [
                SystemMessage(
                    content=[
                        {"type": "text", "text": """You are an expert fashion analyst. Analyze the clothing item in the image and extract detailed information about its properties. 
                        Focus on identifying the category, colors, fabric type, pattern, fit, and other relevant fashion attributes.
                        Be specific and accurate in your analysis. If certain attributes are not clearly visible, make reasonable inferences based on what you can see.
                        Output the information in the specified JSON format."""}
                    ]
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Analyze this clothing item and extract its properties."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"}
                    ]
                )
            ]
            
            response = self.structured_llm.invoke(prompt)
            return response
        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None

    def add_to_wardrobe(self, clothing_item: ClothingItem):
        """Add analyzed clothing item to user's wardrobe"""
        item_dict = json.loads(clothing_item.model_dump_json())
        item_dict['id'] = len(self.wardrobe) + 1
        self.wardrobe.append(item_dict)
        return item_dict['id']

    def get_outfit_recommendations(self, user_preferences: str, num_recommendations: int = 3):
        """Get outfit recommendations based on user preferences"""
        if not self.wardrobe:
            return None
        
        try:
            prompt = [
                SystemMessage(
                    content=[
                        {"type": "text", "text": f"""You are an expert fashion stylist. Based on the user's preferences and their wardrobe items, 
                        recommend complete outfits that match their needs. Consider color coordination, style compatibility, occasion appropriateness, and seasonal suitability.
                        
                        User's Wardrobe:
                        {json.dumps(self.wardrobe, indent=2)}
                        
                        Guidelines:
                        1. Recommend complete outfits (try to include both top and bottom wear when applicable)
                        2. Consider color harmony and style coherence
                        3. Match the occasion and season specified by the user
                        4. Provide practical styling advice
                        5. Maximum {num_recommendations} outfit recommendations
                        6. Only use item IDs that exist in the wardrobe"""}
                    ]
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": f"User preferences: {user_preferences}"}
                    ]
                )
            ]
            
            response = self.recommendation_llm.invoke(prompt)
            return response
        except Exception as e:
            st.error(f"Error getting recommendations: {str(e)}")
            return None

    def get_item_by_id(self, item_id: str):
        """Get wardrobe item by ID"""
        for item in self.wardrobe:
            if str(item['id']) == str(item_id):
                return item
        return None

    def display_outfit_recommendation(self, recommendation: OutfitRecommendation):
        """Display outfit recommendation in a formatted way"""
        st.subheader("🎯 Recommended Outfit")
        
        # Display recommended items
        st.write("**Outfit Items:**")
        for item_id in recommendation.recommended_items:
            item = self.get_item_by_id(item_id)
            if item:
                with st.expander(f"Item {item_id}: {item['category']} - {', '.join(item['color'])}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Description:** {item['description']}")
                        st.write(f"**Fabric:** {item['fabric']}")
                        st.write(f"**Pattern:** {item['pattern']}")
                        st.write(f"**Fit:** {item['fit']}")
                    with col2:
                        st.write(f"**Occasions:** {', '.join(item['occasion'])}")
                        st.write(f"**Seasons:** {', '.join(item['season'])}")
                        st.write(f"**Features:** {', '.join(item['features'])}")
        
        # Display reasoning
        st.write("**Why this outfit works:**")
        st.write(recommendation.reasoning)
        
        # Display styling tips
        if recommendation.style_tips:
            st.write("**Styling Tips:**")
            for tip in recommendation.style_tips:
                st.write(f"• {tip}")

def main():
    st.set_page_config(
        page_title="StyleSync - AI Fashion Assistant",
        page_icon="👗",
        layout="wide"
    )
    
    st.title("👗 StyleSync - AI Fashion Assistant")
    st.markdown("Upload your clothing images and get personalized outfit recommendations!")
    
    # Initialize session state
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'wardrobe_items' not in st.session_state:
        st.session_state.wardrobe_items = []
    
    # Sidebar for API key and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Try to get API key from environment first, then user input
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            api_key = st.text_input("Enter your Gemini API Key:", type="password", 
                                   help="Get your API key from Google AI Studio")
        else:
            st.success("✅ API Key loaded from environment")
        
        if api_key:
            if st.session_state.bot is None:
                try:
                    st.session_state.bot = StyleSyncBot(api_key)
                    st.success("✅ Bot initialized!")
                except Exception as e:
                    st.error(f"Error initializing bot: {str(e)}")
        
        st.header("📊 Wardrobe Stats")
        if st.session_state.bot and st.session_state.bot.wardrobe:
            wardrobe_df = pd.DataFrame(st.session_state.bot.wardrobe)
            st.write(f"**Total Items:** {len(st.session_state.bot.wardrobe)}")
            
            # Category distribution
            if not wardrobe_df.empty:
                category_counts = wardrobe_df['category'].value_counts()
                st.write("**Categories:**")
                for category, count in category_counts.items():
                    st.write(f"• {category}: {count}")
        else:
            st.write("No items in wardrobe yet")
    
    if not api_key:
        st.warning("Please enter your Gemini API key in the sidebar to continue.")
        st.info("💡 You can get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)")
        return
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📷 Add Clothing", "👔 Get Recommendations", "👗 My Wardrobe"])
    
    with tab1:
        st.header("📷 Add Clothing to Your Wardrobe")
        
        uploaded_files = st.file_uploader(
            "Upload clothing images", 
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg'],
            help="Upload clear images of individual clothing items"
        )
        
        if uploaded_files and st.session_state.bot:
            for uploaded_file in uploaded_files:
                with st.expander(f"Analyzing: {uploaded_file.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display image
                        image = Image.open(uploaded_file)
                        st.image(image, caption=uploaded_file.name, use_column_width=True)
                    
                    with col2:
                        if st.button(f"Analyze {uploaded_file.name}", key=f"analyze_{uploaded_file.name}"):
                            with st.spinner("Analyzing clothing item..."):
                                # Get image bytes
                                image_bytes = uploaded_file.getvalue()
                                
                                # Analyze the image
                                clothing_item = st.session_state.bot.analyze_clothing_image(image_bytes)
                                
                                if clothing_item:
                                    # Add to wardrobe
                                    item_id = st.session_state.bot.add_to_wardrobe(clothing_item)
                                    
                                    st.success(f"✅ Added to wardrobe as Item #{item_id}")
                                    
                                    # Display analysis results
                                    st.json(json.loads(clothing_item.model_dump_json()))
    
    with tab2:
        st.header("👔 Get Personalized Recommendations")
        
        if not st.session_state.bot or not st.session_state.bot.wardrobe:
            st.warning("Please add some clothing items to your wardrobe first!")
        else:
            # User preferences input
            col1, col2 = st.columns(2)
            
            with col1:
                occasion = st.selectbox("Occasion", [
                    "Casual", "Work/Professional", "Party", "Date Night", 
                    "Beach/Pool", "Gym/Athletic", "Formal Event", "Travel"
                ])
                
                season = st.selectbox("Season", [
                    "Spring", "Summer", "Fall", "Winter", "Any"
                ])
            
            with col2:
                time_of_day = st.selectbox("Time of Day", [
                    "Morning", "Afternoon", "Evening", "Night", "Any"
                ])
                
                style_preference = st.selectbox("Style Preference", [
                    "Comfortable", "Stylish", "Professional", "Trendy", 
                    "Classic", "Minimalist", "Bold"
                ])
            
            additional_notes = st.text_area(
                "Additional preferences or requirements:",
                placeholder="e.g., prefer bright colors, need pockets, avoid tight fits..."
            )
            
            # Combine preferences
            user_preferences = f"""
            Occasion: {occasion}
            Season: {season}
            Time of Day: {time_of_day}
            Style Preference: {style_preference}
            Additional Notes: {additional_notes}
            """
            
            if st.button("🎯 Get Recommendations", type="primary"):
                with st.spinner("Creating your perfect outfit..."):
                    recommendations = st.session_state.bot.get_outfit_recommendations(user_preferences)
                    
                    if recommendations:
                        st.session_state.bot.display_outfit_recommendation(recommendations)
                    else:
                        st.error("Could not generate recommendations. Please try again.")
    
    with tab3:
        st.header("👗 My Wardrobe")
        
        if st.session_state.bot and st.session_state.bot.wardrobe:
            # Display wardrobe items
            for item in st.session_state.bot.wardrobe:
                with st.expander(f"Item #{item['id']}: {item['category']} - {', '.join(item['color'])}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Category:** {item['category']}")
                        st.write(f"**Colors:** {', '.join(item['color'])}")
                        st.write(f"**Pattern:** {item['pattern']}")
                        st.write(f"**Fabric:** {item['fabric']}")
                    
                    with col2:
                        st.write(f"**Fit:** {item['fit']}")
                        st.write(f"**Sleeve Length:** {item['sleeve_length']}")
                        st.write(f"**Neck Type:** {item['neck_type']}")
                        st.write(f"**Gender:** {item['gender']}")
                    
                    with col3:
                        st.write(f"**Occasions:** {', '.join(item['occasion'])}")
                        st.write(f"**Seasons:** {', '.join(item['season'])}")
                        st.write(f"**Features:** {', '.join(item['features'])}")
                    
                    st.write(f"**Description:** {item['description']}")
            
            # Export wardrobe
            if st.button("📥 Export Wardrobe as JSON"):
                wardrobe_json = json.dumps(st.session_state.bot.wardrobe, indent=2)
                st.download_button(
                    label="Download Wardrobe JSON",
                    data=wardrobe_json,
                    file_name="my_wardrobe.json",
                    mime="application/json"
                )
        else:
            st.info("Your wardrobe is empty. Start by adding some clothing items!")

if __name__ == "__main__":
    # Set port for Railway deployment
    port = int(os.environ.get("PORT", 8501))
    main()