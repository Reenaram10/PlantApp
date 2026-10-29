import os
from image_faiss_engine import ImageFAISSRAG

def main():
    print("====================================")
    print("      IMAGE TO FAISS INDEX BUILDER  ")
    print("====================================")

    # The directories where images are stored
    base_dir = os.path.dirname(os.path.abspath(__file__))
    directories_to_index = [
        os.path.join(base_dir, "plantpics"),
        os.path.join(base_dir, "static", "images", "plants")
    ]

    print("Directories to scan:")
    for d in directories_to_index:
        print(f" - {d}")

    # Initialize the engine
    image_rag = ImageFAISSRAG()

    # Build the FAISS vectors from the images
    image_rag.build_index(directories_to_index)

    print("\n✅ Finished compiling images into vectors!")

if __name__ == "__main__":
    main()
