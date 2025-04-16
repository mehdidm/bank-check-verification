import uuid

# In-memory storage for API keys and permissions (replace with a database in production)
api_keys = {}

def generate_api_key(permissions):
    """Generates a new API key with the given permissions."""
    key = str(uuid.uuid4())
    api_keys[key] = permissions
    return key

def validate_api_key(key, required_permission=None):
    """Validates an API key and optionally checks for a specific permission."""
    if key not in api_keys:
        return False
    if required_permission:
        return required_permission in api_keys[key]
    return True

# Example usage:
# To be removed later or moved to a separate script
if __name__ == "__main__":
    # Generate a key with training permission
    training_key = generate_api_key(['train'])
    print(f"Generated training key: {training_key}")

    # Validate the key
    is_valid = validate_api_key(training_key)
    print(f"Is the key valid? {is_valid}")

    # Check for training permission
    has_permission = validate_api_key(training_key, 'train')
    print(f"Does the key have training permission? {has_permission}")

    # Check for a different permission
    has_other_permission = validate_api_key(training_key, 'deploy')
    print(f"Does the key have deploy permission? {has_other_permission}")