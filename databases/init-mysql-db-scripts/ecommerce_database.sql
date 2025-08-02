-- Create the database
CREATE DATABASE IF NOT EXISTS ecommerce_db;
USE ecommerce_db;

-- Drop tables if they exist (for clean recreation)
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS addresses;
DROP TABLE IF EXISTS product_suppliers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS categories;
DROP TABLE IF EXISTS customers;

-- Table 1: customers
CREATE TABLE customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: categories
CREATE TABLE categories (
    category_id INT AUTO_INCREMENT PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 3: suppliers
CREATE TABLE suppliers (
    supplier_id INT AUTO_INCREMENT PRIMARY KEY,
    supplier_name VARCHAR(100) NOT NULL,
    contact_person VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table 4: products
CREATE TABLE products (
    product_id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    category_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (category_id) REFERENCES categories(category_id) ON DELETE SET NULL
);

-- Table 5: product_suppliers (many-to-many relationship)
CREATE TABLE product_suppliers (
    product_id INT,
    supplier_id INT,
    supply_price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id, supplier_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id) ON DELETE CASCADE
);

-- Table 6: addresses
CREATE TABLE addresses (
    address_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    address_type ENUM('billing', 'shipping') NOT NULL,
    street_address VARCHAR(200) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    country VARCHAR(50) NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- Table 7: orders
CREATE TABLE orders (
    order_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT,
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(12, 2) NOT NULL,
    status ENUM('pending', 'processing', 'shipped', 'delivered', 'cancelled') DEFAULT 'pending',
    shipping_address_id INT,
    billing_address_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE SET NULL,
    FOREIGN KEY (shipping_address_id) REFERENCES addresses(address_id) ON DELETE SET NULL,
    FOREIGN KEY (billing_address_id) REFERENCES addresses(address_id) ON DELETE SET NULL
);

-- Table 8: order_items
CREATE TABLE order_items (
    order_item_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    product_id INT,
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total_price DECIMAL(12, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE SET NULL
);

-- Table 9: payments
CREATE TABLE payments (
    payment_id INT AUTO_INCREMENT PRIMARY KEY,
    order_id INT,
    payment_method ENUM('credit_card', 'debit_card', 'paypal', 'bank_transfer') NOT NULL,
    payment_status ENUM('pending', 'completed', 'failed', 'refunded') DEFAULT 'pending',
    amount DECIMAL(12, 2) NOT NULL,
    transaction_id VARCHAR(100),
    payment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES orders(order_id) ON DELETE CASCADE
);

-- Table 10: reviews
CREATE TABLE reviews (
    review_id INT AUTO_INCREMENT PRIMARY KEY,
    product_id INT,
    customer_id INT,
    rating INT CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    review_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
);

-- INSERT STATEMENTS TO POPULATE THE TABLES

-- Insert customers
INSERT INTO customers (first_name, last_name, email, phone) VALUES
('John', 'Doe', 'john.doe@email.com', '+1-555-0101'),
('Jane', 'Smith', 'jane.smith@email.com', '+1-555-0102'),
('Mike', 'Johnson', 'mike.johnson@email.com', '+1-555-0103'),
('Sarah', 'Williams', 'sarah.williams@email.com', '+1-555-0104'),
('David', 'Brown', 'david.brown@email.com', '+1-555-0105');

-- Insert categories
INSERT INTO categories (category_name, description) VALUES
('Electronics', 'Electronic devices and gadgets'),
('Clothing', 'Apparel and fashion items'),
('Books', 'Physical and digital books'),
('Home & Garden', 'Home improvement and garden supplies'),
('Sports', 'Sports equipment and accessories');

-- Insert suppliers
INSERT INTO suppliers (supplier_name, contact_person, email, phone, address) VALUES
('TechCorp Inc.', 'Alice Manager', 'alice@techcorp.com', '+1-555-1001', '123 Tech Street, Silicon Valley, CA'),
('Fashion Plus', 'Bob Stevens', 'bob@fashionplus.com', '+1-555-1002', '456 Fashion Ave, New York, NY'),
('BookWorld', 'Carol Reader', 'carol@bookworld.com', '+1-555-1003', '789 Library Lane, Boston, MA'),
('HomeSupply Co.', 'Dan Builder', 'dan@homesupply.com', '+1-555-1004', '321 Hardware Blvd, Chicago, IL'),
('SportGear Ltd.', 'Eve Athletic', 'eve@sportgear.com', '+1-555-1005', '654 Fitness Road, Denver, CO');

-- Insert products
INSERT INTO products (product_name, description, price, stock_quantity, category_id) VALUES
('Smartphone X1', 'Latest model smartphone with advanced features', 799.99, 50, 1),
('Wireless Headphones', 'Bluetooth noise-canceling headphones', 199.99, 75, 1),
('Cotton T-Shirt', 'Comfortable 100% cotton t-shirt', 24.99, 100, 2),
('Denim Jeans', 'Classic blue denim jeans', 89.99, 60, 2),
('Programming Guide', 'Complete guide to modern programming', 49.99, 30, 3),
('Mystery Novel', 'Bestselling mystery thriller', 14.99, 40, 3),
('Garden Hose', '50ft expandable garden hose', 39.99, 25, 4),
('Power Drill', 'Cordless power drill with accessories', 129.99, 20, 4),
('Basketball', 'Official size basketball', 29.99, 35, 5),
('Running Shoes', 'Lightweight running shoes', 119.99, 45, 5);

-- Insert product_suppliers relationships
INSERT INTO product_suppliers (product_id, supplier_id, supply_price) VALUES
(1, 1, 650.00), (2, 1, 150.00),
(3, 2, 18.00), (4, 2, 65.00),
(5, 3, 35.00), (6, 3, 10.00),
(7, 4, 28.00), (8, 4, 95.00),
(9, 5, 20.00), (10, 5, 85.00);

-- Insert addresses
INSERT INTO addresses (customer_id, address_type, street_address, city, state, postal_code, country, is_default) VALUES
(1, 'billing', '123 Main St', 'Anytown', 'CA', '12345', 'USA', TRUE),
(1, 'shipping', '123 Main St', 'Anytown', 'CA', '12345', 'USA', TRUE),
(2, 'billing', '456 Oak Ave', 'Springfield', 'NY', '67890', 'USA', TRUE),
(2, 'shipping', '789 Pine St', 'Springfield', 'NY', '67891', 'USA', FALSE),
(3, 'billing', '321 Elm Dr', 'Riverside', 'TX', '54321', 'USA', TRUE),
(3, 'shipping', '321 Elm Dr', 'Riverside', 'TX', '54321', 'USA', TRUE),
(4, 'billing', '654 Maple Ln', 'Hilltown', 'FL', '98765', 'USA', TRUE),
(4, 'shipping', '654 Maple Ln', 'Hilltown', 'FL', '98765', 'USA', TRUE),
(5, 'billing', '987 Cedar Ct', 'Lakeside', 'WA', '13579', 'USA', TRUE),
(5, 'shipping', '987 Cedar Ct', 'Lakeside', 'WA', '13579', 'USA', TRUE);

-- Insert orders
INSERT INTO orders (customer_id, total_amount, status, shipping_address_id, billing_address_id) VALUES
(1, 824.98, 'delivered', 2, 1),
(2, 139.98, 'shipped', 4, 3),
(3, 179.97, 'processing', 6, 5),
(4, 49.99, 'delivered', 8, 7),
(1, 29.99, 'pending', 2, 1);

-- Insert order_items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 799.99, 799.99),
(1, 3, 1, 24.99, 24.99),
(2, 4, 1, 89.99, 89.99),
(2, 6, 1, 14.99, 14.99),
(2, 3, 1, 24.99, 24.99),
(3, 2, 1, 199.99, 199.99),
(4, 5, 1, 49.99, 49.99),
(5, 9, 1, 29.99, 29.99);

-- Insert payments
INSERT INTO payments (order_id, payment_method, payment_status, amount, transaction_id) VALUES
(1, 'credit_card', 'completed', 824.98, 'TXN001234567'),
(2, 'paypal', 'completed', 139.98, 'PP987654321'),
(3, 'debit_card', 'completed', 179.97, 'TXN002345678'),
(4, 'credit_card', 'completed', 49.99, 'TXN003456789'),
(5, 'credit_card', 'pending', 29.99, 'TXN004567890');

-- Insert reviews
INSERT INTO reviews (product_id, customer_id, rating, review_text) VALUES
(1, 1, 5, 'Amazing smartphone! Fast, reliable, and great camera quality.'),
(3, 1, 4, 'Good quality t-shirt, fits well and comfortable.'),
(4, 2, 5, 'Perfect fit jeans, excellent quality denim.'),
(6, 2, 4, 'Great mystery novel, kept me engaged throughout.'),
(2, 3, 5, 'Excellent headphones, great sound quality and noise cancellation.'),
(5, 4, 5, 'Very comprehensive programming guide, highly recommended.'),
(9, 5, 4, 'Good quality basketball, perfect for outdoor games.');

-- Display table relationships summary
SELECT 'Database created successfully with the following table relationships:' as Status;
SELECT 'customers -> addresses (1:many)' as Relationship
UNION SELECT 'customers -> orders (1:many)'
UNION SELECT 'customers -> reviews (1:many)'
UNION SELECT 'categories -> products (1:many)'
UNION SELECT 'products -> order_items (1:many)'
UNION SELECT 'products -> reviews (1:many)'
UNION SELECT 'products <-> suppliers (many:many via product_suppliers)'
UNION SELECT 'orders -> order_items (1:many)'
UNION SELECT 'orders -> payments (1:many)'
UNION SELECT 'addresses -> orders (1:many for shipping/billing)';