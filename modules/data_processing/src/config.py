INDEX_BODY = {
    'settings': {
        'index': {
            'knn': True  # Enable k-NN for vector similarity search
        }
    },
    'mappings': {
        'properties': {
            'embedding': {
                'type': 'knn_vector',
                'dimension': None # Will be set later dynamically
            },
            'text': {
                'type': 'text'
            },
            'metadata': {
                'properties': {
                    'content_type': {'type': 'keyword'},
                    'url': {
                        'type': 'nested',
                        'properties': {
                            'format': {'type': 'keyword'},
                            'platform': {'type': 'keyword'},
                            'value': {'type': 'keyword'}
                        }
                    },
                    'title': {'type': 'text'},
                    'publication_name': {'type': 'text'},
                    'doi': {'type': 'keyword'},
                    'publication_date': {'type': 'date', 'format': 'yyyy-MM-dd'},
                    'starting_page': {'type': 'integer'},
                    'ending_page': {'type': 'integer'},
                    'open_access': {'type': 'boolean'},
                    'abstract': {
                        'properties': {
                            'h1': {'type': 'text'},
                            'p': {'type': 'text'}
                        }
                    },
                    'section': {'type': 'text'}
                }
            }
        }
    }
}