-- ==============================================================================
-- PlantApp - Microsoft SQL Server Database Schema Setup
-- Database: PlantAppDB
-- Supports multi-tenancy, complete relational integrity, cascading deletes,
-- composite indexing, and full compatibility with SQL Server 2017+ / 2019 / 2022 / 2025.
-- ==============================================================================

-- 1. Ensure Database Exists
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'PlantAppDB')
BEGIN
    CREATE DATABASE PlantAppDB;
    PRINT 'Database PlantAppDB created.';
END
GO

USE PlantAppDB;
GO

-- 2. Multi-Tenant Master: businesses
IF OBJECT_ID(N'dbo.businesses', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.businesses (
        business_id     VARCHAR(50)     NOT NULL,
        business_name   VARCHAR(100)    NOT NULL,
        owner_name      VARCHAR(100)    NULL,
        email           VARCHAR(100)    NULL,
        phone           VARCHAR(20)     NULL,
        address         NVARCHAR(MAX)   NULL,
        city            VARCHAR(50)     NULL,
        state           VARCHAR(50)     NULL,
        country         VARCHAR(50)     NULL,
        status          VARCHAR(20)     NULL DEFAULT 'Active',
        created_at      DATETIME        NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME        NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_businesses PRIMARY KEY CLUSTERED (business_id)
    );
    PRINT 'Table businesses created.';
END
GO

-- 3. Users and Authentication: login
IF OBJECT_ID(N'dbo.login', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.login (
        id              INT IDENTITY(1,1)   NOT NULL,
        username        VARCHAR(100)        NULL,
        password        VARCHAR(200)        NULL,
        email           VARCHAR(120)        COLLATE SQL_Latin1_General_CP1_CS_AS NOT NULL,
        last_login      DATETIME            NULL,
        last_logout     DATETIME            NULL,
        is_online       BIT                 NULL DEFAULT 0,
        role            VARCHAR(20)         NULL DEFAULT 'NURSERY_ADMIN',
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        CONSTRAINT PK_login PRIMARY KEY CLUSTERED (id),
        CONSTRAINT UQ_login_email UNIQUE (email)
    );
    PRINT 'Table login created.';
END
GO

-- 4. Supplier Master: supplier
IF OBJECT_ID(N'dbo.supplier', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.supplier (
        supplier_id     INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        supplier_name   VARCHAR(100)        NOT NULL,
        contact_person  VARCHAR(100)        NULL,
        phone           VARCHAR(20)         NULL,
        email           VARCHAR(100)        NULL,
        address         NVARCHAR(MAX)       NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_supplier PRIMARY KEY CLUSTERED (supplier_id)
    );
    PRINT 'Table supplier created.';
END
GO

-- 5. Plant Catalog: plant
IF OBJECT_ID(N'dbo.plant', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.plant (
        plant_id        INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        plant_name      VARCHAR(100)        NOT NULL,
        description     NVARCHAR(MAX)       NULL,
        price           FLOAT               NOT NULL,
        image_path      VARCHAR(255)        NULL,
        stock           INT                 NULL DEFAULT 0,
        discount_price  FLOAT               NULL DEFAULT 0.0,
        CONSTRAINT PK_plant PRIMARY KEY CLUSTERED (plant_id)
    );
    PRINT 'Table plant created.';
END
GO

-- 6. Category Master: category
IF OBJECT_ID(N'dbo.category', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.category (
        category_id     INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        category_name   VARCHAR(100)        NOT NULL,
        CONSTRAINT PK_category PRIMARY KEY CLUSTERED (category_id)
    );
    PRINT 'Table category created.';
END
GO

-- 7. Plant Categories Junction: plant_category
IF OBJECT_ID(N'dbo.plant_category', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.plant_category (
        plant_category_id   INT IDENTITY(1,1)   NOT NULL,
        plant_id            INT                 NOT NULL,
        category_id         INT                 NOT NULL,
        CONSTRAINT PK_plant_category PRIMARY KEY CLUSTERED (plant_category_id),
        CONSTRAINT FK_plant_category_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE,
        CONSTRAINT FK_plant_category_category FOREIGN KEY (category_id) REFERENCES dbo.category(category_id) ON DELETE CASCADE,
        CONSTRAINT UQ_plant_category UNIQUE (plant_id, category_id)
    );
    PRINT 'Table plant_category created.';
END
GO

-- 8. Plant Varieties: variety
IF OBJECT_ID(N'dbo.variety', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.variety (
        variety_id          INT IDENTITY(1,1)   NOT NULL,
        business_id         VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        plant_id            INT                 NOT NULL,
        variety_name        VARCHAR(120)        NOT NULL,
        variety_description NVARCHAR(MAX)       NULL,
        variety_price       FLOAT               NULL,
        created_at          DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_variety PRIMARY KEY CLUSTERED (variety_id),
        CONSTRAINT FK_variety_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE
    );
    PRINT 'Table variety created.';
END
GO

-- 9. Plant Synonyms: plant_synonym
IF OBJECT_ID(N'dbo.plant_synonym', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.plant_synonym (
        synonym_id  INT IDENTITY(1,1)   NOT NULL,
        plant_id    INT                 NULL,
        synonym     NVARCHAR(MAX)       NOT NULL,
        created_at  DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_plant_synonym PRIMARY KEY CLUSTERED (synonym_id),
        CONSTRAINT FK_plant_synonym_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE
    );
    PRINT 'Table plant_synonym created.';
END
GO

-- 10. Inventory Transactions: inventory_transaction
IF OBJECT_ID(N'dbo.inventory_transaction', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.inventory_transaction (
        transaction_id  INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        plant_id        INT                 NOT NULL,
        supplier_id     INT                 NULL,
        type            VARCHAR(10)         NOT NULL, -- 'ADD' or 'REMOVE'
        quantity        INT                 NOT NULL,
        notes           NVARCHAR(MAX)       NULL,
        bill_no         VARCHAR(50)         NULL,
        bill_date       VARCHAR(20)         NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_inventory_transaction PRIMARY KEY CLUSTERED (transaction_id),
        CONSTRAINT FK_inv_trans_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE,
        CONSTRAINT FK_inv_trans_supplier FOREIGN KEY (supplier_id) REFERENCES dbo.supplier(supplier_id) ON DELETE SET NULL
    );
    PRINT 'Table inventory_transaction created.';
END
GO

-- 11. Supplier Purchases: supplier_purchase
IF OBJECT_ID(N'dbo.supplier_purchase', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.supplier_purchase (
        purchase_id     INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        supplier_id     INT                 NOT NULL,
        bill_no         VARCHAR(50)         NOT NULL,
        bill_date       DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        total_amount    FLOAT               NOT NULL,
        amount_paid     FLOAT               NULL DEFAULT 0.0,
        balance         FLOAT               NOT NULL,
        status          VARCHAR(20)         NULL DEFAULT 'Pending',
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_supplier_purchase PRIMARY KEY CLUSTERED (purchase_id),
        CONSTRAINT FK_supp_purchase_supplier FOREIGN KEY (supplier_id) REFERENCES dbo.supplier(supplier_id) ON DELETE CASCADE
    );
    PRINT 'Table supplier_purchase created.';
END
GO

-- 12. Purchase Items: purchase_item
IF OBJECT_ID(N'dbo.purchase_item', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.purchase_item (
        item_id         INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        purchase_id     INT                 NOT NULL,
        plant_id        INT                 NOT NULL,
        quantity        INT                 NOT NULL,
        unit_price      FLOAT               NOT NULL,
        CONSTRAINT PK_purchase_item PRIMARY KEY CLUSTERED (item_id),
        CONSTRAINT FK_purchase_item_purchase FOREIGN KEY (purchase_id) REFERENCES dbo.supplier_purchase(purchase_id) ON DELETE CASCADE,
        CONSTRAINT FK_purchase_item_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE
    );
    PRINT 'Table purchase_item created.';
END
GO

-- 13. Shopping Cart: cart_new
IF OBJECT_ID(N'dbo.cart_new', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.cart_new (
        cart_id         INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        user_id         VARCHAR(100)        NOT NULL,
        plant_id        INT                 NOT NULL,
        quantity        INT                 NULL DEFAULT 1,
        added_at        DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_cart_new PRIMARY KEY CLUSTERED (cart_id),
        CONSTRAINT FK_cart_new_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE
    );
    PRINT 'Table cart_new created.';
END
GO

-- 14. Customer Orders: orders
IF OBJECT_ID(N'dbo.orders', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.orders (
        order_id        INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        plant_name      VARCHAR(100)        NULL,
        user_id         INT                 NULL,
        order_date      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        total_amount    FLOAT               NULL DEFAULT 0.0,
        payment_status  VARCHAR(50)         NULL DEFAULT 'Pending',
        order_group_id  VARCHAR(100)        NULL,
        quantity        INT                 NULL DEFAULT 1,
        payment_method  VARCHAR(50)         NULL,
        order_status    VARCHAR(50)         NULL DEFAULT 'Processing',
        shipping_address NVARCHAR(MAX)      NULL,
        delivery_date   VARCHAR(50)         NULL,
        tracking_number VARCHAR(100)        NULL,
        notes           NVARCHAR(MAX)       NULL,
        price           FLOAT               NULL DEFAULT 0.0,
        amount_paid     FLOAT               NULL DEFAULT 0.0,
        balance         FLOAT               NULL DEFAULT 0.0,
        CONSTRAINT PK_orders PRIMARY KEY CLUSTERED (order_id),
        CONSTRAINT FK_orders_user FOREIGN KEY (user_id) REFERENCES dbo.login(id) ON DELETE SET NULL
    );
    PRINT 'Table orders created.';
END
GO

-- 15. Category Synonyms: category_synonym
IF OBJECT_ID(N'dbo.category_synonym', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.category_synonym (
        synonym_id      INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        category_id     INT                 NOT NULL,
        synonym         VARCHAR(100)        NOT NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_category_synonym PRIMARY KEY CLUSTERED (synonym_id),
        CONSTRAINT FK_cat_synonym_category FOREIGN KEY (category_id) REFERENCES dbo.category(category_id) ON DELETE CASCADE
    );
    PRINT 'Table category_synonym created.';
END
GO

-- 16. Queries Master: query
IF OBJECT_ID(N'dbo.query', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.query (
        query_id        INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        user_id         INT                 NULL,
        description     NVARCHAR(MAX)       NOT NULL,
        canonical_key   VARCHAR(100)        NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_query PRIMARY KEY CLUSTERED (query_id),
        CONSTRAINT FK_query_user FOREIGN KEY (user_id) REFERENCES dbo.login(id) ON DELETE CASCADE
    );
    PRINT 'Table query created.';
END
GO

-- 17. Query Results: query_result
IF OBJECT_ID(N'dbo.query_result', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.query_result (
        result_id           INT IDENTITY(1,1)   NOT NULL,
        business_id         VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        query_id            INT                 NOT NULL,
        result_description  NVARCHAR(MAX)       NOT NULL,
        result_source       VARCHAR(50)         NOT NULL,
        response_type       VARCHAR(50)         NULL DEFAULT 'text',
        response_data       NVARCHAR(MAX)       NULL,
        created_at          DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_query_result PRIMARY KEY CLUSTERED (result_id),
        CONSTRAINT FK_query_result_query FOREIGN KEY (query_id) REFERENCES dbo.query(query_id) ON DELETE CASCADE
    );
    PRINT 'Table query_result created.';
END
GO

-- 18. User Search Queries: user_queries
IF OBJECT_ID(N'dbo.user_queries', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.user_queries (
        uq_id           INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        user_id         INT                 NOT NULL,
        plant_id        INT                 NULL,
        query           NVARCHAR(MAX)       NOT NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_user_queries PRIMARY KEY CLUSTERED (uq_id),
        CONSTRAINT FK_user_queries_user FOREIGN KEY (user_id) REFERENCES dbo.login(id) ON DELETE CASCADE,
        CONSTRAINT FK_user_queries_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE SET NULL
    );
    PRINT 'Table user_queries created.';
END
GO

-- 19. User Interest & Leads: user_interest
IF OBJECT_ID(N'dbo.user_interest', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.user_interest (
        interest_id     INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        user_id         INT                 NOT NULL,
        plant_id        INT                 NOT NULL,
        interest_type   VARCHAR(20)         NOT NULL,
        notes           NVARCHAR(MAX)       NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_user_interest PRIMARY KEY CLUSTERED (interest_id),
        CONSTRAINT FK_user_interest_user FOREIGN KEY (user_id) REFERENCES dbo.login(id) ON DELETE CASCADE,
        CONSTRAINT FK_user_interest_plant FOREIGN KEY (plant_id) REFERENCES dbo.plant(plant_id) ON DELETE CASCADE
    );
    PRINT 'Table user_interest created.';
END
GO

-- 20. User Category Interest: user_category_interest
IF OBJECT_ID(N'dbo.user_category_interest', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.user_category_interest (
        id                  INT IDENTITY(1,1)   NOT NULL,
        user_id             INT                 NULL,
        category_id         INT                 NULL,
        interest_type       VARCHAR(30)         NULL,
        count               INT                 NULL DEFAULT 1,
        last_interaction    DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_user_category_interest PRIMARY KEY CLUSTERED (id),
        CONSTRAINT FK_uci_user FOREIGN KEY (user_id) REFERENCES dbo.login(id) ON DELETE CASCADE,
        CONSTRAINT FK_uci_category FOREIGN KEY (category_id) REFERENCES dbo.category(category_id) ON DELETE CASCADE
    );
    PRINT 'Table user_category_interest created.';
END
GO

-- 21. User Chat History: conversation
IF OBJECT_ID(N'dbo.conversation', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.conversation (
        conversation_id INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        user_id         INT                 NULL,
        conversation    NVARCHAR(MAX)       NOT NULL,
        created_at      DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT PK_conversation PRIMARY KEY CLUSTERED (conversation_id),
        CONSTRAINT FK_conversation_user FOREIGN KEY (user_id) REFERENCES dbo.login(id) ON DELETE CASCADE
    );
    PRINT 'Table conversation created.';
END
GO

-- 22. Cached LLM Responses: llm_results
IF OBJECT_ID(N'dbo.llm_results', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.llm_results (
        id                  INT IDENTITY(1,1)   NOT NULL,
        business_id         VARCHAR(50)         NULL DEFAULT 'NURS-0001',
        user_query          NVARCHAR(MAX)       NOT NULL,
        llm_response        NVARCHAR(MAX)       NOT NULL,
        category_suggested  VARCHAR(100)        NULL,
        plants_found        INT                 NULL DEFAULT 0,
        llm_name            VARCHAR(50)         NULL,
        created_at          DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at          DATETIME            NULL,
        CONSTRAINT PK_llm_results PRIMARY KEY CLUSTERED (id)
    );
    PRINT 'Table llm_results created.';
END
GO

-- 23. Plant Botanical Knowledge: plant_knowledge
IF OBJECT_ID(N'dbo.plant_knowledge', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.plant_knowledge (
        id                  INT IDENTITY(1,1)   NOT NULL,
        plant_name          VARCHAR(255)        NULL,
        scientific_name     VARCHAR(255)        NULL,
        description         NVARCHAR(MAX)       NULL,
        sunlight            NVARCHAR(MAX)       NULL,
        water_requirement   NVARCHAR(MAX)       NULL,
        soil_type           NVARCHAR(MAX)       NULL,
        fertilizer          NVARCHAR(MAX)       NULL,
        medicinal_uses      NVARCHAR(MAX)       NULL,
        common_diseases     NVARCHAR(MAX)       NULL,
        care_tips           NVARCHAR(MAX)       NULL,
        CONSTRAINT PK_plant_knowledge PRIMARY KEY CLUSTERED (id)
    );
    PRINT 'Table plant_knowledge created.';
END
GO

-- 24. Disease Diagnostic Knowledge: disease_knowledge
IF OBJECT_ID(N'dbo.disease_knowledge', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.disease_knowledge (
        id                  INT IDENTITY(1,1)   NOT NULL,
        disease_name        VARCHAR(255)        NULL,
        symptoms            NVARCHAR(MAX)       NULL,
        causes              NVARCHAR(MAX)       NULL,
        organic_treatment   NVARCHAR(MAX)       NULL,
        chemical_treatment  NVARCHAR(MAX)       NULL,
        prevention          NVARCHAR(MAX)       NULL,
        CONSTRAINT PK_disease_knowledge PRIMARY KEY CLUSTERED (id)
    );
    PRINT 'Table disease_knowledge created.';
END
GO

-- 25. Security & Multi-Tenant Audit Logs: audit_logs
IF OBJECT_ID(N'dbo.audit_logs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.audit_logs (
        id              INT IDENTITY(1,1)   NOT NULL,
        business_id     VARCHAR(50)         NULL,
        user_id         INT                 NULL,
        user_email      VARCHAR(100)        NULL,
        action          VARCHAR(100)        NOT NULL,
        module          VARCHAR(50)         NOT NULL,
        record_id       VARCHAR(50)         NULL,
        timestamp       DATETIME            NULL DEFAULT CURRENT_TIMESTAMP,
        details         NVARCHAR(MAX)       NULL,
        CONSTRAINT PK_audit_logs PRIMARY KEY CLUSTERED (id)
    );
    PRINT 'Table audit_logs created.';
END
GO

-- ==============================================================================
-- 26. Multi-Tenant Composite Indexes for Query Acceleration & Isolation
-- ==============================================================================

-- Index Helper Macro
IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_businesses_status' AND object_id = OBJECT_ID('dbo.businesses'))
    CREATE NONCLUSTERED INDEX idx_businesses_status ON dbo.businesses (status);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_login_business_id' AND object_id = OBJECT_ID('dbo.login'))
    CREATE NONCLUSTERED INDEX idx_login_business_id ON dbo.login (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_login_email' AND object_id = OBJECT_ID('dbo.login'))
    CREATE NONCLUSTERED INDEX idx_login_email ON dbo.login (email);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_plant_business_id' AND object_id = OBJECT_ID('dbo.plant'))
    CREATE NONCLUSTERED INDEX idx_plant_business_id ON dbo.plant (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_plant_biz_plant' AND object_id = OBJECT_ID('dbo.plant'))
    CREATE NONCLUSTERED INDEX idx_plant_biz_plant ON dbo.plant (business_id, plant_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_plant_biz_name' AND object_id = OBJECT_ID('dbo.plant'))
    CREATE NONCLUSTERED INDEX idx_plant_biz_name ON dbo.plant (business_id, plant_name);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_orders_business_id' AND object_id = OBJECT_ID('dbo.orders'))
    CREATE NONCLUSTERED INDEX idx_orders_business_id ON dbo.orders (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_orders_biz_order' AND object_id = OBJECT_ID('dbo.orders'))
    CREATE NONCLUSTERED INDEX idx_orders_biz_order ON dbo.orders (business_id, order_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_orders_biz_date' AND object_id = OBJECT_ID('dbo.orders'))
    CREATE NONCLUSTERED INDEX idx_orders_biz_date ON dbo.orders (business_id, order_date);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_orders_biz_user' AND object_id = OBJECT_ID('dbo.orders'))
    CREATE NONCLUSTERED INDEX idx_orders_biz_user ON dbo.orders (business_id, user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_orders_biz_status' AND object_id = OBJECT_ID('dbo.orders'))
    CREATE NONCLUSTERED INDEX idx_orders_biz_status ON dbo.orders (business_id, order_status);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_orders_group_id' AND object_id = OBJECT_ID('dbo.orders'))
    CREATE NONCLUSTERED INDEX idx_orders_group_id ON dbo.orders (order_group_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_inventory_business_id' AND object_id = OBJECT_ID('dbo.inventory_transaction'))
    CREATE NONCLUSTERED INDEX idx_inventory_business_id ON dbo.inventory_transaction (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_inventory_biz_plant' AND object_id = OBJECT_ID('dbo.inventory_transaction'))
    CREATE NONCLUSTERED INDEX idx_inventory_biz_plant ON dbo.inventory_transaction (business_id, plant_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_inventory_biz_date' AND object_id = OBJECT_ID('dbo.inventory_transaction'))
    CREATE NONCLUSTERED INDEX idx_inventory_biz_date ON dbo.inventory_transaction (business_id, created_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_supplier_business_id' AND object_id = OBJECT_ID('dbo.supplier'))
    CREATE NONCLUSTERED INDEX idx_supplier_business_id ON dbo.supplier (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_supplier_biz' AND object_id = OBJECT_ID('dbo.supplier'))
    CREATE NONCLUSTERED INDEX idx_supplier_biz ON dbo.supplier (business_id, supplier_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_supplier_purchase_business_id' AND object_id = OBJECT_ID('dbo.supplier_purchase'))
    CREATE NONCLUSTERED INDEX idx_supplier_purchase_business_id ON dbo.supplier_purchase (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_supplier_purchase_biz' AND object_id = OBJECT_ID('dbo.supplier_purchase'))
    CREATE NONCLUSTERED INDEX idx_supplier_purchase_biz ON dbo.supplier_purchase (business_id, supplier_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_purchase_item_business_id' AND object_id = OBJECT_ID('dbo.purchase_item'))
    CREATE NONCLUSTERED INDEX idx_purchase_item_business_id ON dbo.purchase_item (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_purchase_item_biz' AND object_id = OBJECT_ID('dbo.purchase_item'))
    CREATE NONCLUSTERED INDEX idx_purchase_item_biz ON dbo.purchase_item (business_id, purchase_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_variety_business_id' AND object_id = OBJECT_ID('dbo.variety'))
    CREATE NONCLUSTERED INDEX idx_variety_business_id ON dbo.variety (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_cart_business_id' AND object_id = OBJECT_ID('dbo.cart_new'))
    CREATE NONCLUSTERED INDEX idx_cart_business_id ON dbo.cart_new (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_cart_biz_user' AND object_id = OBJECT_ID('dbo.cart_new'))
    CREATE NONCLUSTERED INDEX idx_cart_biz_user ON dbo.cart_new (business_id, user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_category_business_id' AND object_id = OBJECT_ID('dbo.category'))
    CREATE NONCLUSTERED INDEX idx_category_business_id ON dbo.category (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_category_synonym_business_id' AND object_id = OBJECT_ID('dbo.category_synonym'))
    CREATE NONCLUSTERED INDEX idx_category_synonym_business_id ON dbo.category_synonym (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_category_synonym_biz' AND object_id = OBJECT_ID('dbo.category_synonym'))
    CREATE NONCLUSTERED INDEX idx_category_synonym_biz ON dbo.category_synonym (business_id, category_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_llm_results_business_id' AND object_id = OBJECT_ID('dbo.llm_results'))
    CREATE NONCLUSTERED INDEX idx_llm_results_business_id ON dbo.llm_results (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_llm_results_biz' AND object_id = OBJECT_ID('dbo.llm_results'))
    CREATE NONCLUSTERED INDEX idx_llm_results_biz ON dbo.llm_results (business_id, created_at);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_query_business_id' AND object_id = OBJECT_ID('dbo.query'))
    CREATE NONCLUSTERED INDEX idx_query_business_id ON dbo.query (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_query_result_business_id' AND object_id = OBJECT_ID('dbo.query_result'))
    CREATE NONCLUSTERED INDEX idx_query_result_business_id ON dbo.query_result (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_user_queries_business_id' AND object_id = OBJECT_ID('dbo.user_queries'))
    CREATE NONCLUSTERED INDEX idx_user_queries_business_id ON dbo.user_queries (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_user_queries_biz_user' AND object_id = OBJECT_ID('dbo.user_queries'))
    CREATE NONCLUSTERED INDEX idx_user_queries_biz_user ON dbo.user_queries (business_id, user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_user_interest_business_id' AND object_id = OBJECT_ID('dbo.user_interest'))
    CREATE NONCLUSTERED INDEX idx_user_interest_business_id ON dbo.user_interest (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_user_interest_biz_user' AND object_id = OBJECT_ID('dbo.user_interest'))
    CREATE NONCLUSTERED INDEX idx_user_interest_biz_user ON dbo.user_interest (business_id, user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_business_id' AND object_id = OBJECT_ID('dbo.conversation'))
    CREATE NONCLUSTERED INDEX idx_conversation_business_id ON dbo.conversation (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_conversation_biz_user' AND object_id = OBJECT_ID('dbo.conversation'))
    CREATE NONCLUSTERED INDEX idx_conversation_biz_user ON dbo.conversation (business_id, user_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_audit_logs_business_id' AND object_id = OBJECT_ID('dbo.audit_logs'))
    CREATE NONCLUSTERED INDEX idx_audit_logs_business_id ON dbo.audit_logs (business_id);

IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'idx_audit_logs_biz' AND object_id = OBJECT_ID('dbo.audit_logs'))
    CREATE NONCLUSTERED INDEX idx_audit_logs_biz ON dbo.audit_logs (business_id, timestamp);
GO

-- ==============================================================================
-- 27. Seed Essential Multi-Tenant Records & System Administrator
-- ==============================================================================

-- Default System Organization
IF NOT EXISTS (SELECT 1 FROM dbo.businesses WHERE business_id = 'SYSTEM')
BEGIN
    INSERT INTO dbo.businesses (business_id, business_name, owner_name, email, phone, status)
    VALUES ('SYSTEM', 'Platform System Administration', 'System Root', 'admin@system.com', '0000000000', 'Active');
    PRINT 'Seeded SYSTEM business entity.';
END

-- Primary Nursery Tenant
IF NOT EXISTS (SELECT 1 FROM dbo.businesses WHERE business_id = 'NURS-0001')
BEGIN
    INSERT INTO dbo.businesses (business_id, business_name, owner_name, email, phone, status)
    VALUES ('NURS-0001', 'Green Valley Nursery', 'Green Valley Admin', 'admin@plantapp.com', '1234567890', 'Active');
    PRINT 'Seeded NURS-0001 business entity.';
END

-- Super Admin User
IF NOT EXISTS (SELECT 1 FROM dbo.login WHERE email = 'admin@system.com' OR role = 'SUPER_ADMIN')
BEGIN
    INSERT INTO dbo.login (username, email, password, role, business_id, is_online)
    VALUES ('Super Admin', 'admin@system.com', 'admin123', 'SUPER_ADMIN', 'SYSTEM', 0);
    PRINT 'Seeded Super Admin user.';
END

-- Default Nursery Admin User
IF NOT EXISTS (SELECT 1 FROM dbo.login WHERE email = 'admin@plantapp.com')
BEGIN
    INSERT INTO dbo.login (username, email, password, role, business_id, is_online)
    VALUES ('Green Valley Admin', 'admin@plantapp.com', 'admin123', 'NURSERY_ADMIN', 'NURS-0001', 0);
    PRINT 'Seeded Nursery Admin user.';
END

-- Default App Admin User
IF NOT EXISTS (SELECT 1 FROM dbo.login WHERE email = 'admin@gmail.com')
BEGIN
    INSERT INTO dbo.login (username, email, password, role, business_id, is_online)
    VALUES ('admin', 'admin@gmail.com', 'admin@123', 'NURSERY_ADMIN', 'NURS-0001', 0);
    PRINT 'Seeded App Admin user.';
END
GO

PRINT '=======================================================';
PRINT '✅ PlantAppDB Microsoft SQL Server schema setup complete!';
PRINT '=======================================================';
GO
