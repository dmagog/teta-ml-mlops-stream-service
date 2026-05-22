.PHONY: up down reset logs ps test train seed

up:        ## собрать и поднять стек
	docker compose up --build -d

down:      ## остановить стек
	docker compose down

reset:     ## полный перезапуск с очисткой данных
	docker compose down -v && docker compose up --build -d

logs:      ## смотреть логи
	docker compose logs -f --tail=100

ps:        ## статус контейнеров
	docker compose ps

test:      ## юнит-тесты сервиса
	cd scorer && python -m pytest -q

train:     ## сгенерировать данные и обучить модель
	python model/generate_data.py && python model/train.py

seed:      ## подать 1000 транзакций в Kafka (headless)
	python scripts/produce.py --limit 1000 --rate 100
