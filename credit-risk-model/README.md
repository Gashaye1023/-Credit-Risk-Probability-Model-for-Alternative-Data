# Credit Risk Model

## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?
The Basel II Accord emphasizes the importance of risk measurement for financial institutions, mandating that banks maintain adequate capital reserves based on the risks they undertake. This requirement necessitates the development of interpretable and well-documented models to ensure transparency and regulatory compliance. An interpretable model allows stakeholders to understand the rationale behind credit decisions, facilitating better risk management and adherence to regulatory standards.

### Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?
Creating a proxy variable is essential because it acts as a substitute for the direct "default" label, allowing us to categorize customers into risk segments based on their behaviors. However, relying on a proxy variable introduces potential risks, including misclassification of customers, which can lead to inappropriate loan approvals or denials. If the proxy does not accurately reflect true risk, the bank may face financial losses or reputational damage.

### What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?
Using a simple, interpretable model like Logistic Regression offers clarity and ease of understanding, which is crucial in a regulated environment. However, it may lack the predictive power of more complex models like Gradient Boosting, which can capture intricate patterns in the data. The trade-off lies in balancing interpretability with performance; while complex models may improve accuracy, they can complicate compliance efforts and make it harder to explain decisions to regulators and customers.