-- 与 genshin_ai 库示例一致：users + characters 通过 uid（varchar）关联。

CREATE TABLE IF NOT EXISTS users (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  uid VARCHAR(20) NOT NULL,
  resin INT DEFAULT 120,
  goal VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS characters (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  uid VARCHAR(20),
  name VARCHAR(50),
  level INT,
  talent_level VARCHAR(20)
);
