# Predicción de Trayectorias de Misiles con Redes Neuronales (Newton y Newton-Barrier)

Este repositorio está inspirado en el paper:

> **Applying Newton Algorithms within a Supervised Feed Forward Neural Network Architecture to Forecast a Missile Trajectory**  
> *Tarek A. Tutunji, 10th International Conference on Aerospace Sciences & Aviation Technology, 2003*

---

## Resumen del Paper

El artículo propone una **arquitectura de red neuronal** para predecir la trayectoria de un misil.  
En lugar de entrenar con el método clásico de **backpropagation con gradiente descendente**, se aplican **métodos tipo Newton** y **Newton-Barrier**, logrando una convergencia más rápida y estable.  

### Puntos clave

1. **Redes neuronales para predicción de trayectorias**  
   - Se usan redes neuronales feed-forward para predecir la **posición futura** de un misil.  
   - Entradas: posiciones pasadas.  
   - Salidas: próxima posición \((x, y)\).  

2. **Algoritmo de Newton**  
   - Convergencia más rápida que el gradiente descendente.  
   - Usa información de segunda orden (matriz Hessiana o aproximaciones).  
   - Implementado con el **Recursive Prediction Error Method (RPEM)**.  

3. **Método Newton-Barrier**  
   - Añade **restricciones a los pesos** para evitar inestabilidad numérica.  
   - Se usa una **función de barrera logarítmica** como método de punto interior:  
     \[
     \Phi(w) = E(w) - \mu \sum_i [\log(M - w_i) + \log(M + w_i)]
     \]  
   - Mantiene los pesos en el rango seguro \([-M, M]\).  

4. **Modelos de trayectoria**  
   - **Misil balístico (sin propulsión):**  
     \[
     x(t) = v_0 \cos(\alpha)\, t
     \]  
     \[
     y(t) = v_0 \sin(\alpha)\, t - \tfrac{1}{2} g t^2
     \]
   - **Misil con thruster:**  
     \[
     y(t) = v_0 \sin(\alpha)\, t - \tfrac{1}{2} g t^2 + T(t)
     \]  
     donde \(T(t)\) es la fuerza de empuje adicional.  

5. **Resultados de simulación**  
   - **Newton:** rápida convergencia en trayectorias simples.  
   - **Newton-Barrier:** mejor desempeño en trayectorias complejas con thruster.  
   - Aplicación práctica:  
     **Radar mide → Red neuronal predice → Sistema interceptor actúa**.  

---

## 🧠 Lo que implementa este repositorio

1. **Simulación de trayectorias balísticas**  
   - Ecuaciones físicas básicas.  
   - Variaciones con thruster para trayectorias más realistas.  

2. **Datos con ruido (simulación de radar)**  
   - Se añade ruido gaussiano a las trayectorias reales.  

3. **Red neuronal feed-forward**  
   - Entrenada en modo offline con datos simulados.  
   - Predicciones en tiempo real usando ventanas de posiciones pasadas.  

4. **Entrenamiento con restricciones (Newton-Barrier)**  
   - Función de pérdida con barrera logarítmica.  
   - Mantiene los pesos en un rango seguro y mejora la estabilidad.  

5. **Comparación gráfica**  
   - Trayectoria real (sin ruido).  
   - Trayectoria medida (radar con ruido).  
   - Predicción de la red neuronal:  
     - Newton simple.  
     - Newton-Barrier.  

---

## Resultados esperados

- Gráficas mostrando:  
  - Trayectoria real del misil.  
  - Medición de radar con ruido.  
  - Predicción de la red neuronal.  

Se observa que **Newton-Barrier** obtiene predicciones más estables cuando hay empuje adicional (thruster).  

