import os
import json # For parsing JSON from request
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure the Google Gemini API
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file or environment variables.")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Using a recent flash model
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- Helper function to construct the prompt for Local Businesses ---
def construct_local_biz_caption_prompt(data):
    """Constructs a detailed prompt for the Gemini API based on local business form data."""
    business_name = data.get('businessName') or "Our business"
    business_type = data.get('businessType', 'local business')
    
    post_type = data.get('postType', 'general announcement').replace('_', ' ')
    key_message = data.get('keyMessage', 'something exciting!')
    tone = data.get('tone', 'friendly & casual').replace('_', ' ')
    call_to_action = data.get('callToAction', 'no specific cta')
    include_emojis = data.get('includeEmojis') == 'on'
    # Ensure numVariations is correctly fetched for this form
    num_variations = int(data.get('numVariations', 3))


    prompt_lines = [
        f"You are an expert social media manager specializing in engaging content for diverse local businesses.",
        f"Generate {num_variations} distinct social media caption(s).",
        f"The business type is: '{business_type}'.",
        f"The desired tone for the caption(s) is '{tone}'.",
        f"The purpose of the post is: '{post_type}'.",
        f"Key message/details to include: '{key_message}'.",
    ]

    if business_name != "Our business":
        prompt_lines.append(f"The business name is '{business_name}'.")

    if call_to_action != "no specific cta" and call_to_action != "custom_cta":
        cta_text_map = {
            "visit_us": "Visit Us Today!",
            "shop_now_online": "Shop Now/Order Online!",
            "book_appointment_now": "Book Your Appointment Now!",
            "learn_more": "Learn More (link in bio/DM us)!",
            "tag_friend": "Tag a Friend Who Needs This!",
            "contact_us": "Contact Us for Details!"
        }
        cta_text = cta_text_map.get(call_to_action, call_to_action.replace('_', ' ').title() + "!")
        prompt_lines.append(f"Include a call to action like: '{cta_text}'.")
    elif call_to_action == "custom_cta":
        prompt_lines.append("The user wants a custom call to action, infer it from the key message or make a general one if not clear.")

    if include_emojis:
        prompt_lines.append("Please include relevant emojis to make the caption(s) engaging.")
    else:
        prompt_lines.append("Do not use any emojis in the caption(s).")

    prompt_lines.extend([
        "Instructions for the AI:",
        f"- Tailor the caption(s) specifically to the nature of a '{business_type}'. Make it sound authentic for that type of business.",
        "- Craft engaging captions suitable for platforms like Instagram, Facebook, or X (formerly Twitter).",
        "- If a product or service is mentioned, make it sound appealing and highlight its benefits.",
        "- If an offer, event, or announcement is mentioned, ensure the details are clear and enticing.",
        "- Naturally weave in the desired tone and call to action (if specified).",
        "- Provide distinct caption options. Each caption should be a complete thought.",
        "- Do not include hashtags unless specifically asked for in the key message.",
        "- Focus solely on generating the caption text. Do not add any introductory or concluding remarks, or labels like 'Caption 1:'.",
        "- Each caption should be on a new line if multiple are generated."
    ])
    return "\n".join(prompt_lines)


# --- NEW: Helper function to construct the prompt for Artisan Product Descriptions ---
def construct_artisan_description_prompt(data):
    """Constructs a detailed prompt for the Gemini API for artisan product descriptions."""
    creator_name = data.get('creatorName')
    product_name = data.get('productName', 'this unique item') # Required in form
    product_category = data.get('productCategory', 'handmade product') # Required in form
    key_materials = data.get('keyMaterials', 'quality materials') # Required in form
    creation_process = data.get('creationProcess')
    inspiration = data.get('inspiration')
    unique_selling_points = data.get('uniqueSellingPoints', 'it is special') # Required in form
    artisan_tone = data.get('artisanTone', 'story_driven_evocative').replace('_', ' ')
    # Ensure numVariations is correctly fetched for this form (name attribute is 'numVariations')
    num_variations = int(data.get('numVariations', 2)) # Default to 2 as in HTML

    prompt_lines = [
        f"You are an expert copywriter specializing in crafting compelling and unique product descriptions for artisans and handmade sellers.",
        f"Generate {num_variations} distinct product description(s) for an artisan product.",
        f"The product is: '{product_name}', a type of '{product_category}'.",
        f"It is made primarily from: '{key_materials}'.",
        f"The desired tone for the description(s) is '{artisan_tone}'.",
        f"Key unique selling points and customer benefits are: '{unique_selling_points}'.",
    ]

    if creator_name:
        prompt_lines.append(f"The creator/brand name is '{creator_name}'.")
    if creation_process:
        prompt_lines.append(f"Highlights of the creation process/technique: '{creation_process}'.")
    if inspiration:
        prompt_lines.append(f"The inspiration or story behind the product is: '{inspiration}'.")

    prompt_lines.extend([
        "Instructions for the AI:",
        f"- Write descriptions that are evocative and highlight the uniqueness of a handmade '{product_category}'.",
        "- Emphasize the craftsmanship, materials, and the story/inspiration if provided.",
        "- Appeal to customers looking for unique, high-quality artisan goods.",
        "- Keep the descriptions suitable for an online shop listing (e.g., Etsy, personal website).",
        "- Provide distinct description options. Each description should be a complete thought and well-structured for readability.",
        "- Focus solely on generating the product description text. Do not add any introductory or concluding remarks, or labels like 'Description 1:'.",
        "- Each description should be on a new line if multiple are generated, and be a paragraph or two in length."
    ])
    return "\n".join(prompt_lines)


# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/generate_local_biz_captions', methods=['POST'])
def generate_local_biz_captions():
    """Handles form submission for local businesses, interacts with Gemini API, and returns generated captions."""
    if not model:
        return jsonify({"error": "Gemini API model not configured. Please check server logs."}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided in request."}), 400
        prompt = construct_local_biz_caption_prompt(data)
        print("---- Constructed Local Biz Prompt ----")
        print(prompt)
        print("------------------------------------")
        response = model.generate_content(prompt)
        generated_text = ""
        captions = []
        if response.candidates and response.candidates[0].content.parts:
            generated_text = response.candidates[0].content.parts[0].text
            captions = [caption.strip() for caption in generated_text.split('\n') if caption.strip()]
        elif hasattr(response, 'text') and response.text: # Fallback
            generated_text = response.text
            captions = [caption.strip() for caption in generated_text.split('\n') if caption.strip()]
        
        print("---- Gemini API Response Text (Local Biz) ----")
        print(generated_text)
        print("--------------------------------------------")
        print("---- Parsed Captions (Local Biz) ----")
        print(captions)
        print("-------------------------------------")

        num_variations_requested = int(data.get('numVariations', 3))
        final_captions = []
        if captions:
            if len(captions) >= num_variations_requested:
                final_captions = captions[:num_variations_requested]
            else:
                final_captions = captions 
        
        if not final_captions:
            error_message = "The AI could not generate captions. Please try rephrasing or adding more detail."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                for rating in response.prompt_feedback.safety_ratings:
                    if rating.category.name != "HARM_CATEGORY_UNSPECIFIED" and rating.probability.name not in ["NEGLIGIBLE", "LOW"]:
                        error_message = f"Content generation blocked due to safety concerns ({rating.category.name}). Please revise your input."
                        break
            return jsonify({"error": error_message}), 400
        return jsonify({"captions": final_captions})
    except Exception as e:
        print(f"Error during local biz caption generation: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


# --- NEW: Flask Route for Artisan Product Descriptions ---
@app.route('/generate_artisan_description', methods=['POST'])
def generate_artisan_description():
    """Handles form submission for artisan product descriptions, interacts with Gemini API, and returns generated descriptions."""
    if not model:
        return jsonify({"error": "Gemini API model not configured. Please check server logs."}), 500
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided in request."}), 400
        
        prompt = construct_artisan_description_prompt(data)
        print("---- Constructed Artisan Prompt ----") # For debugging
        print(prompt)
        print("----------------------------------")

        response = model.generate_content(prompt)
        generated_text = ""
        descriptions = []

        if response.candidates and response.candidates[0].content.parts:
            generated_text = response.candidates[0].content.parts[0].text
            # Product descriptions might be longer and could be paragraph-based.
            # Splitting by double newline might be better if AI formats them as paragraphs.
            # For now, sticking to single newline split, assuming AI gives distinct descriptions on new lines.
            descriptions = [desc.strip() for desc in generated_text.split('\n') if desc.strip()]
        elif hasattr(response, 'text') and response.text: # Fallback
            generated_text = response.text
            descriptions = [desc.strip() for desc in generated_text.split('\n') if desc.strip()]

        print("---- Gemini API Response Text (Artisan) ----")
        print(generated_text)
        print("--------------------------------------------")
        print("---- Parsed Descriptions (Artisan) ----")
        print(descriptions)
        print("---------------------------------------")

        num_variations_requested = int(data.get('numVariations', 2)) # Default from HTML form for artisan
        final_descriptions = []

        if descriptions:
            if len(descriptions) >= num_variations_requested:
                final_descriptions = descriptions[:num_variations_requested]
            else: # AI gave fewer than requested
                final_descriptions = descriptions
        
        if not final_descriptions:
            error_message = "The AI could not generate descriptions. Please try rephrasing or adding more detail."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                for rating in response.prompt_feedback.safety_ratings:
                    if rating.category.name != "HARM_CATEGORY_UNSPECIFIED" and rating.probability.name not in ["NEGLIGIBLE", "LOW"]:
                        error_message = f"Content generation blocked due to safety concerns ({rating.category.name}). Please revise your input."
                        break
            return jsonify({"error": error_message}), 400
        
        # IMPORTANT: The JavaScript expects a key named "descriptions" for this endpoint
        return jsonify({"descriptions": final_descriptions})

    except Exception as e:
        print(f"Error during artisan description generation: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
