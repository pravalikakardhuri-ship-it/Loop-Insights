"""
data/generate_sample_data.py
-----------------------------
Generates a synthetic customer feedback dataset for LOOP Insights.

Produces >= 1000 rows covering positive, negative, and neutral feedback
across a realistic mix of products and categories. No real personal
information is used -- customer names are drawn from a generic first-name
pool paired with a generated ID.

Run:
    python data/generate_sample_data.py

Output:
    data/sample_feedback.csv
"""

import csv
import random
import sys
from datetime import date, timedelta
from pathlib import Path

random.seed(42)  # reproducible dataset

OUTPUT_PATH = Path(__file__).resolve().parent / "sample_feedback.csv"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import config  # noqa: E402

PRODUCT_TO_CATEGORY = config.product_category_map()
ALL_PRODUCTS = list(PRODUCT_TO_CATEGORY)

# Templates that only make sense for electronics-style products (batteries,
# screens, setup, noise...). They are skipped for fashion, beauty, etc.
ELECTRONICS_CATEGORIES = {"Electronics", "Accessories", "Wearables"}
ELECTRONICS_ONLY_WORDS = (
    "battery", "display", "overheat", "set up", "setup", "interface", "noise",
    "sound", "gadget", "manual", "older model", "screen",
)

FIRST_NAMES = [
    "Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohan", "Kavya",
    "Arjun", "Divya", "Karan", "Neha", "Sanjay", "Pooja", "Aditya", "Riya",
    "Manoj", "Shreya", "Vivek", "Ananya", "Suresh", "Meera", "Rajesh", "Isha",
    "Nikhil", "Tanvi", "Deepak", "Simran", "Ashok", "Ritu", "Gaurav", "Nisha",
    "Harish", "Swati", "Manish", "Preeti", "Sunil", "Kritika", "Ravi", "Aarti",
]

# --- Feedback templates -----------------------------------------------------
# Each template is a (text, rating_range, category_hint) tuple.
# category_hint is used only to keep rating/sentiment internally consistent.

POSITIVE_TEMPLATES = [
    "The {product} exceeded my expectations, great build quality.",
    "Really happy with this {product}, works perfectly.",
    "Delivery was super fast, got my {product} a day early.",
    "Excellent customer service, they resolved my {product} issue quickly.",
    "Great value for money, this {product} is worth every rupee.",
    "The {product} quality is outstanding, highly recommend.",
    "Fast shipping and the {product} arrived in perfect condition.",
    "Customer support was very helpful when I had questions about the {product}.",
    "Best {product} I've purchased in this price range.",
    "The packaging was neat and the {product} works flawlessly.",
    "Very satisfied with the {product}, will buy again.",
    "Impressive performance from the {product}, no complaints at all.",
    "Support team responded within minutes and solved my {product} query.",
    "Good value, the {product} performs better than expected for the price.",
    "The {product} looks premium and functions perfectly.",
    "I've been using my {product} daily and it hasn't let me down once.",
    "Setup was quick and the {product} just worked right out of the box.",
    "Battery life on the {product} is way better than I expected.",
    "The build feels sturdy and well thought out for a {product} at this price.",
    "Ordered on a whim and honestly this {product} became my favorite gadget.",
    "Everything about the buying experience for this {product} felt smooth.",
    "My family loves the new {product}, no regrets purchasing it.",
    "The {product} does exactly what it promises, and does it well.",
    "Warranty support was hassle-free when I reached out about the {product}.",
    "Genuinely impressed by how well this {product} performs day to day.",
    "The {product} was easy to set up and the manual was actually helpful.",
    "Sound and display quality on this {product} are top notch for the price.",
    "Couldn't be happier, the {product} arrived early and works great.",
    "This {product} has held up well after months of regular use.",
    "The interface on the {product} is intuitive, even for a first-time user.",
    "This {product} is good, does what I need.",
    "Good {product}, works nicely and looks great too.",
    "Great {product}, I love it.",
    "Nice {product}, happy with the purchase.",
    "This {product} is really good, no issues so far.",
]

NEGATIVE_TEMPLATES = [
    "My {product} order arrived three days late, very disappointing.",
    "The {product} quality is poor, it stopped working within a week.",
    "This {product} is overpriced for what you actually get.",
    "Customer support never responded to my {product} complaint.",
    "Requested a refund for the {product} but it's been weeks with no update.",
    "The app kept crashing when I tried to track my {product} order.",
    "Delivery was delayed by almost a week, no communication from the courier.",
    "The {product} arrived damaged and packaging was flimsy.",
    "Terrible experience, the {product} broke on the second day of use.",
    "Refund process for the {product} is confusing and slow.",
    "Customer service was rude when I called about my {product}.",
    "The {product} is not worth the price, quality feels cheap.",
    "App crashed twice while checking out my {product} order.",
    "Never received updates about my late {product} delivery.",
    "The {product} stopped working and support hasn't helped at all.",
    "Payment was deducted twice for the same {product} order.",
    "Website glitched during checkout and I got charged extra for the {product}.",
    "Battery on the {product} drains within a couple of hours, quite frustrating.",
    "The {product} started making a strange noise after just a few uses.",
    "Packaging looked like it had been opened before it even reached me.",
    "I waited on hold for over an hour just to ask about my {product}.",
    "Return window closed before support even replied to my {product} query.",
    "The {product} doesn't match what was shown in the listing photos.",
    "Setup instructions for the {product} were confusing and incomplete.",
    "Support kept transferring my call instead of resolving my {product} issue.",
    "The {product} developed a crack near the seams after light use.",
    "I was charged shipping fees that were never mentioned at checkout.",
    "This {product} feels like a downgrade compared to the older model.",
    "Tracking showed my {product} as delivered, but it never arrived.",
    "The {product} overheats after only a short time of normal use.",
    "This {product} is bad, would not recommend.",
    "Bad {product}, broke almost immediately.",
    "Awful {product}, total waste of money.",
    "Worst {product} I've ever bought, avoid this.",
    "This {product} is terrible, very bad quality.",
    "Really bad experience with this {product}, would not buy again.",
]

NEUTRAL_TEMPLATES = [
    "The {product} is okay, does what it's supposed to do.",
    "Average experience overall with the {product}.",
    "The {product} works as expected, nothing special.",
    "Delivery was on time, the {product} is decent.",
    "It's a fine {product}, meets basic expectations.",
    "The {product} is fine for casual use, nothing exceptional.",
    "Delivery was normal, the {product} matches the description.",
    "Neither impressed nor disappointed with the {product}.",
    "The {product} is functional but could use some improvements.",
    "Standard {product}, does the job without any issues.",
    "It took a moment to get used to the {product}, but it's manageable now.",
    "The {product} is reasonable for the price, nothing more, nothing less.",
    "Packaging was simple and the {product} arrived as described.",
    "The {product} performs adequately for everyday tasks.",
    "I have mixed feelings about the {product}, some parts are good, some aren't.",
    "The {product} looks similar to what was advertised, works fine so far.",
    "Nothing stood out about the {product}, but nothing went wrong either.",
    "The {product} arrived on schedule and matches its description.",
    "Would call this a middle-of-the-road {product}, gets the job done.",
    "The {product} is usable but I probably wouldn't buy the same model again.",
]

# --- Category-specific templates ---------------------------------------------
# Keyed by category. Each has positive / negative / neutral lists. Phrases are
# chosen so the keyword lexicon in training/train_topic.py labels them with the
# intended topic.

CATEGORY_TEMPLATES = {
    "Fashion": {
        "pos": [
            "The {product} fits perfectly and the fabric quality is excellent.",
            "Stitching on this {product} is neat and it feels durable.",
            "Love the colour of the {product}, it looks exactly like the photos.",
            "The size chart for the {product} was accurate, ordered with confidence.",
        ],
        "neg": [
            "The {product} stitching came apart after two washes.",
            "Wrong size delivered for the {product} and the size chart is misleading.",
            "The colour of the {product} faded after the first wash.",
            "The fabric quality of the {product} feels cheap and rough.",
        ],
        "neu": [
            "The {product} fits okay, the size chart was roughly right.",
            "Decent {product}, colour is slightly different from the photo.",
        ],
    },
    "Home & Kitchen": {
        "pos": [
            "The {product} is easy to assemble and works quietly.",
            "Easy to clean and very sturdy, this {product} is a good buy.",
            "The {product} saves me so much time in the kitchen every day.",
            "Installation of the {product} was simple and the instructions were clear.",
        ],
        "neg": [
            "The assembly instructions for the {product} were confusing and incomplete.",
            "The {product} stopped working after a month, the motor is weak.",
            "Hard to clean and the {product} makes a lot of noise.",
            "A part of the {product} was missing from the box.",
        ],
        "neu": [
            "The {product} does the job, though it is a bit bulky.",
            "Average {product}, easy to assemble but nothing special.",
        ],
    },
    "Beauty & Personal Care": {
        "pos": [
            "The {product} quality is outstanding and the fragrance lasts all day.",
            "My skin feels great after using this {product}, will repurchase.",
            "The {product} works flawlessly and the packaging was neat.",
        ],
        "neg": [
            "The {product} quality is poor and the fragrance faded within an hour.",
            "This {product} caused irritation on my skin, very disappointing.",
            "The {product} bottle arrived damaged and leaking.",
        ],
        "neu": [
            "The {product} is fine, the fragrance is mild.",
            "Average {product}, results are okay after a few weeks.",
        ],
    },
    "Health & Fitness": {
        "pos": [
            "The {product} quality is outstanding and it is great for daily workouts.",
            "Good value, the {product} has held up well after months of use.",
            "The {product} arrived early and works great.",
        ],
        "neg": [
            "The {product} quality is poor and it wore out within weeks.",
            "The seal of the {product} was broken when it arrived damaged.",
            "The {product} is overpriced for the quality you get.",
        ],
        "neu": [
            "The {product} is okay for beginners, nothing special.",
            "Average {product}, does what it says.",
        ],
    },
}

# Availability complaints/praise apply to every category.
AVAILABILITY_TEMPLATES = {
    "pos": [
        "The {product} was back in stock quickly and I managed to order it.",
        "Glad the {product} was back in stock before the festive sale.",
    ],
    "neg": [
        "The {product} was out of stock right after I placed my order.",
        "My {product} order was cancelled because it was sold out.",
        "The {product} has been unavailable for weeks, very frustrating.",
    ],
    "neu": [
        "The {product} was out of stock for a day, then it was fine.",
    ],
}


def _templates_for(category: str, kind: str) -> list:
    """Generic + category-specific + availability templates for a product."""
    base = {"pos": POSITIVE_TEMPLATES, "neg": NEGATIVE_TEMPLATES, "neu": NEUTRAL_TEMPLATES}[kind]
    if category not in ELECTRONICS_CATEGORIES:
        base = [t for t in base if not any(w in t.lower() for w in ELECTRONICS_ONLY_WORDS)]
    specific = CATEGORY_TEMPLATES.get(category, {}).get(kind, [])
    # Repeat category-specific lines so they show up often enough to matter.
    return list(base) + specific * 3 + AVAILABILITY_TEMPLATES[kind]


def _random_date(start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def generate_dataset(num_records: int = 3000) -> list[dict]:
    """Generate `num_records` synthetic feedback rows.

    Roughly targets a 55% positive / 25% negative / 20% neutral split, which
    keeps the sentiment classes reasonably balanced for model training while
    still reflecting a realistic "mostly satisfied, meaningful complaints"
    business.
    """
    start_date = date(2026, 1, 1)
    end_date = date(2026, 8, 1)

    records = []
    for i in range(1, num_records + 1):
        product = random.choice(ALL_PRODUCTS)
        category = PRODUCT_TO_CATEGORY[product]
        customer_name = random.choice(FIRST_NAMES)
        customer_id = f"C{i:04d}"
        feedback_date = _random_date(start_date, end_date)

        roll = random.random()
        if roll < 0.55:
            template = random.choice(_templates_for(category, "pos"))
            rating = random.choices([4, 5], weights=[0.35, 0.65])[0]
        elif roll < 0.80:
            template = random.choice(_templates_for(category, "neg"))
            rating = random.choices([1, 2], weights=[0.55, 0.45])[0]
        else:
            template = random.choice(_templates_for(category, "neu"))
            rating = 3

        feedback_text = template.format(product=product.lower())
        # Capitalize first letter of the sentence.
        feedback_text = feedback_text[0].upper() + feedback_text[1:]

        records.append(
            {
                "customer_id": customer_id,
                "customer_name": customer_name,
                "product": product,
                "category": category,
                "rating": rating,
                "feedback": feedback_text,
                "date": feedback_date.isoformat(),
            }
        )

    return records


def write_csv(records: list[dict], path: Path) -> None:
    fieldnames = ["customer_id", "customer_name", "product", "category", "rating", "feedback", "date"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    data = generate_dataset(3000)
    write_csv(data, OUTPUT_PATH)
    print(f"Generated {len(data)} records -> {OUTPUT_PATH}")
