import weaviate

def list_collections():
    # Connect to local Weaviate instance
    client = weaviate.connect_to_local()
    
    try:
        # Get all collections
        collections = client.collections.list_all()
        
        # Print the collection names
        print("Available collections in local Weaviate:")
        if collections:
            for collection in collections:
                if hasattr(collection, 'name'):
                        print(f"- {collection.name}")
                else:
                    print(f"- {collection}")
        else:
            print("No collections found.")
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
    finally:
        # Close the connection
        client.close()

if __name__ == "__main__":
    list_collections()