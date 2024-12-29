from conn.connect import *
import pandas as pd

# def fetch_and_save_data():
#     # Create a database connection
#     db = DBConnection()
#     conn = db.create_db_connection()
    
#     # SQL query to join products and categories tables
#     query = """
#     SELECT 
#         p.id AS product_id,
#         p.name AS product_name,
#         p.description AS product_description,
#         p.category_id,
#         c.name AS category_name
#     FROM 
#         products p
#     JOIN 
#         categories c 
#     ON 
#         p.category_id = c.id
#     WHERE 
#         p.deleted_at IS NULL  # Assuming products marked deleted should be excluded
#     """

#     # Execute the query and load the results into a DataFrame
#     df = pd.read_sql(query, conn)
    
#     # Drop duplicate rows based on product_name, product_description, and category_name
#     df = df.drop_duplicates(subset=['product_name', 'product_description', 'category_name'])
    
#     # # Save the DataFrame as a CSV file
#     csv_file_path = 'product_data1.csv'
#     df.to_csv(csv_file_path, index=False)
    
#     print(f"Data has been saved to {csv_file_path}")
    
#     return df

def fetch_and_save_data():
    # Create a database connection
    db = DBConnection()
    conn = db.create_db_connection()
    
    # SQL query to join products and categories tables
    query = """
    SELECT 
        p.id AS product_id,
        p.name AS product_name,
        p.description AS product_description,
        p.category_id,
        c.name AS category_name
    FROM 
        products p
    JOIN 
        categories c 
    ON 
        p.category_id = c.id
    WHERE 
        p.deleted_at IS NULL  # Assuming products marked deleted should be excluded
    """

    # Execute the query and load the results into a DataFrame
    df = pd.read_sql(query, conn)
    
    # Save the DataFrame as a CSV file
    csv_file_path = 'product_data.csv'
    df.to_csv(csv_file_path, index=False)
    
    print(f"Data has been saved to {csv_file_path}")
    
    return df
