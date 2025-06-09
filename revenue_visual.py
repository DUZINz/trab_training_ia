from calculations import revenue_per_industry
import matplotlib.pyplot as plt

# Gera os dados de receita média por indústria
revenue_data = revenue_per_industry()

# Seleciona as 25 maiores receitas
top_25_revenue_data = revenue_data.sort_values(ascending=False).head(25)

# Cria o gráfico de barras das 25 maiores indústrias em receita
plt.figure(figsize=(10, 6))
plt.bar(top_25_revenue_data.index, top_25_revenue_data.values)

plt.title("Revenue for Top 25 Industries (sorted by Revenue)")
plt.xlabel("Industry")
plt.ylabel("Revenue")
plt.xticks(rotation=45)

plt.show()