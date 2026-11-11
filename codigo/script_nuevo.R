
#     1. INSTALACIÓN DE PAQUETES Y LECTURA DE DATASET:

# Paquetes que necesitamos para todo el proyecto:
paquetes <- c(
  "tidyverse",       # manipulación de datos y gráficos
  "caret",           # partición, validación cruzada y modelos
  "janitor",         # limpieza de nombres y tablas
  "rpart",           # árboles de clasificación
  "rpart.plot",      # visualización de árboles
  "knitr",           # tablas en Markdown
  "dplyr",
  "Metrics",         # Para calcular RMSE, MAE fácilmente
  "lmtest", 
  "car", 
  "glmnet", 
  "mgcv",
  "caretEnsemble",   # permite combinar varios modelos de caret en uno solo.
  "ranger",          # motor rápido que usa caret para el Random Forest.
  "xgboost"          # motor que usa caret para XGBoost.
)



# Instalamos solo los que falten (se ejecuta una sola vez por máquina)
nuevos <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(nuevos) > 0) {
  install.packages(nuevos)
}

# Cargamos todas las librerías. invisible() evita que se imprima en pantalla
# la lista de TRUE/FALSE que devuelve lapply() al cargar cada una.
invisible(lapply(paquetes, library, character.only = TRUE))

# Se establece la ruta del dataset:
ruta <- "../datos/used_cars.csv"

if (!file.exists(ruta)) {
  stop(
    "No se encontró ", ruta, ". "
  )
}

# Se leen los datos:
datos <- read.csv(ruta, header = TRUE)



# ==============================================================================================================================
# ==============================================================================================================================



#     2. PARTICIÓN TRAIN / TEST (se hace ANTES de cualquier limpieza o transformación):

# Carga y partición de datos:

# Suponiendo que el dataset ya está cargado como 'datos'
# y que la variable respuesta es 'price'

# El split se hace sobre los datos crudos, de modo que ninguna decisión de limpieza
# se calcule mirando el set de test. A partir de aquí, 'test' no se toca
# hasta la evaluación final.

set.seed(123)  # Reproducibilidad

n <- nrow(datos)
indices_train <- sample(1:n, size = 0.8 * n)

train <- datos[indices_train, ]
test  <- datos[-indices_train, ]



# ==============================================================================================================================
# ==============================================================================================================================



#     3. LIMPIEZA DE DATOS:

# Se revisan los primeros detos del conjunto:
head(datos)
str(datos)
dim(datos)

# Valores faltantes:
colSums(is.na(datos))    # No hay valores NA en este dataset

# Valores "unknown" por columna:
sapply(datos, function(x) sum(x == "unknown")) # No hay valores "unknown"

# Contar cadenas vacías por columna
sapply(datos, function(x) {
  if (is.character(x) || is.factor(x)) {
    sum(trimws(as.character(x)) == "", na.rm = TRUE)
  } else {
    0
  }
})

datos[datos == ""] <- NA

# Encontrar el tipo de "guion" presente en las casillas, ya que no es el típico:
datos$ext_col[609]
charToRaw(datos$ext_col[609])

# Contar cadenas vacías por columna
sapply(datos, function(x) {
  if (is.character(x) || is.factor(x)) {
    sum(trimws(as.character(x)) == "–", na.rm = TRUE)
  } else {
    0
  }
})

datos[datos == "–"] <- NA

# Cantidad de NA por columna
colSums(is.na(datos))
# Porcentaje de NA por columna
round(colMeans(is.na(datos)) * 100, 2)

# Se intercambian los NA de la variable NA por "unknown" ya que si se eliminaran las filas con datos nulos, 
# se perdería una cantidad considerable de información:
datos$clean_title[is.na(datos$clean_title)] <- "unknown"
datos$accident[is.na(datos$accident)] <- "unknown"
datos$fuel_type[is.na(datos$fuel_type)] <- "unknown"
datos$engine[is.na(datos$engine)] <- "unknown"
datos$transmission[is.na(datos$transmission)] <- "unknown"
datos$ext_col[is.na(datos$ext_col)] <- "unknown"
datos$int_col[is.na(datos$int_col)] <- "unknown"


# Valores "unknown" por columna:
sapply(datos, function(x) sum(x == "unknown"))

# Duplicados:
sum(duplicated(datos)) # No hay duplicados

# Traspasar datos numéricos a clase numeric:

# Para price y milage, hay que limpiar la variable para que no imprima error por los símbolos:
datos$price <- as.numeric(
  gsub("[$,]", "", trimws(as.character(datos$price)))
)

datos$milage <- as.numeric(
  gsub(",", "", 
       gsub(" mi\\.", "", as.character(datos$milage)))
)

# En el caso del año del modelo, se hace conversión directa:
datos$model_year <- as.numeric(as.character(datos$model_year))

# Rangos razonables:
range(datos$model_year) # 1974 hasta 2024
range(datos$milage)  # 1,000 mi hasta 99,999 mi
range(datos$price)   # $1,599,000 hasta $99,999


# No se eliminan variables, todas sirven


summary(datos)
names(datos)




# ==============================================================================================================================
# ==============================================================================================================================



#     4. BASELINES:

# Carga y partición de datos:

# Suponiendo que el dataset ya está cargado como 'datos'
# y que la variable respuesta es 'price'


# Baseline 1: Media global

media_global <- mean(train$price, na.rm = TRUE)

pred_media <- rep(media_global, nrow(test))

rmse_media <- rmse(test$price, pred_media)
mae_media  <- mae(test$price, pred_media)

cat("Baseline - Media global\n")
cat("RMSE:", rmse_media, "\n")
cat("MAE :", mae_media, "\n\n")

# Baseline 2: Valor previo (persistencia)

# Dado que no existe un orden temporal real en el dataset,
# este baseline se construye utilizando el orden original de las
# observaciones como aproximación. Su interpretación debe tomarse
# con cautela y mencionarse como limitación en el informe.

test_ordenado <- test %>% arrange(row_number())

pred_valor_previo <- lag(test_ordenado$price, n = 1)

# Se elimina el primer valor (NA) para la comparación
real_vp <- test_ordenado$price[-1]
pred_vp <- pred_valor_previo[-1]

rmse_vp <- rmse(real_vp, pred_vp)
mae_vp  <- mae(real_vp, pred_vp)

cat("Baseline - Valor previo (persistencia)\n")
cat("RMSE:", rmse_vp, "\n")
cat("MAE :", mae_vp, "\n\n")



#Baseline 3: Promedio por marca:
promedio_por_marca <- train %>%
  group_by(brand) %>%
  summarise(
    precio_promedio = mean(price, na.rm = TRUE),
    .groups = "drop"
  )

# Unir el promedio de cada marca al conjunto de prueba
test_grupo <- test %>%
  left_join(promedio_por_marca, by = "brand")

# Para marcas no vistas en entrenamiento, usar la media global
test_grupo$precio_promedio[
  is.na(test_grupo$precio_promedio)
] <- media_global

# Calcular métricas
rmse_grupo <- rmse(
  test_grupo$price,
  test_grupo$precio_promedio
)

mae_grupo <- mae(
  test_grupo$price,
  test_grupo$precio_promedio
)

cat("Baseline - Promedio por grupo (marca)\n")
cat("RMSE:", rmse_grupo, "\n")
cat("MAE :", mae_grupo, "\n\n")


#Resumen comparativo de los baselines:
resumen_baselines <- data.frame(
  Modelo = c("Media global", "Valor previo", "Promedio por grupo (marca)"),
  RMSE   = c(rmse_media, rmse_vp, rmse_grupo),
  MAE    = c(mae_media, mae_vp, mae_grupo)
)

print(resumen_baselines) # La baseline 3 es la que tiene menos error




# ==============================================================================================================================
# ==============================================================================================================================



#     5. REGRESIÓN LINEAL MÚLTIPLE(OLS)

# Primero, se deben preparar las variables para los modelos de esta parte del código:

# Variables explicativas, no se consideraron ext_col e int_col. Son variables con muchas categorías y pueden complicar tanto 
# la estimación OLS como el cálculo del VIF(Se intentó en una versión anterior del código, y el VIF dió error por lo mismo)
predictores <- c(
  "brand",
  "model_year",
  "milage",
  "fuel_type",
  "transmission",
  "accident",
  "clean_title"
)

# Conservar solo las variables disponibles
predictores <- intersect(predictores, names(train))

# Convertir variables categóricas a factor
categoricas <- intersect(
  c("brand", "fuel_type", "transmission", "accident",
    "clean_title", "ext_col", "int_col"),
  predictores
)

# Se ha utilizado una herramienta LLM para codificar el siguiente bloque de código que contiene el ciclo for:
for (v in categoricas) {
  train[[v]] <- as.character(train[[v]])
  test[[v]]  <- as.character(test[[v]])
  
  # Categorías disponibles en entrenamiento
  niveles <- unique(train[[v]])
  nivel_frecuente <- names(which.max(table(train[[v]])))
  
  # Categorías de prueba no vistas:
  # usar "unknown" si existe en entrenamiento;
  # en caso contrario, usar la categoría más frecuente.
  reemplazo <- if ("unknown" %in% niveles) {
    "unknown"
  } else {
    nivel_frecuente
  }
  
  test[[v]][
    !is.na(test[[v]]) &
      !(test[[v]] %in% niveles)
  ] <- reemplazo
  
  train[[v]] <- factor(train[[v]], levels = niveles)
  test[[v]]  <- factor(test[[v]], levels = niveles)
}

# Fórmula común para OLS
formula_ols <- reformulate(
  predictores,
  response = "price"
)

modelo_ols <- lm(
  formula_ols,
  data = train,
  na.action = na.exclude
)

summary(modelo_ols)

# Predicciones sobre test
pred_ols <- predict(modelo_ols, newdata = test)

# Métricas
rmse_ols <- Metrics::rmse(test$price, pred_ols)
mae_ols  <- Metrics::mae(test$price, pred_ols)

cat("OLS - RMSE:", rmse_ols, "\n")
cat("OLS - MAE:", mae_ols, "\n")

# Intervalos de confianza de los coeficientes
confint(modelo_ols, level = 0.95)


# DIAGNNÓSTICO DE RESIDUOS

residuos <- residuals(modelo_ols)
ajustados <- fitted(modelo_ols)

# Gráficos de diagnóstico clásicos
par(mfrow = c(2, 2))
plot(modelo_ols)
par(mfrow = c(1, 1))

# Heterocedasticidad: test de Breusch-Pagan
test_bp <- lmtest::bptest(modelo_ols)
print(test_bp)

# Normalidad: gráfico Q-Q
qqnorm(residuos, main = "Q-Q plot de residuos OLS")
qqline(residuos, col = "red")

# Shapiro-Wilk sobre una muestra de residuos
# Es preferible no usar millones de observaciones.
set.seed(123)

residuos_validos <- residuos[is.finite(residuos)]

muestra_residuos <- sample(
  residuos_validos,
  size = min(5000, length(residuos_validos))
)

if (length(muestra_residuos) >= 3) {
  print(shapiro.test(muestra_residuos))
}

# MULTICOLINEALIDAD: VIF / GVIF
vif_ols <- car::vif(modelo_ols)
print(vif_ols)



# ==============================================================================================================================
# ==============================================================================================================================



#     6. MATRICES DE DISEÑO PARA GLMNET

# Crear matrices a partir de los mismos predictores
X_train <- model.matrix(
  formula_ols,
  data = train
)[, -1, drop = FALSE]

X_test <- model.matrix(
  formula_ols,
  data = test
)[, -1, drop = FALSE]

# Asegurar que ambas matrices tengan las mismas columnas
columnas_comunes <- intersect(
  colnames(X_train),
  colnames(X_test)
)

X_train <- X_train[, columnas_comunes, drop = FALSE]
X_test  <- X_test[, columnas_comunes, drop = FALSE]

y_train <- train$price
y_test  <- test$price



# RIDGE: alpha = 0

set.seed(123)

cv_ridge <- glmnet::cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 0,
  family = "gaussian",
  nfolds = 10,
  type.measure = "mse",
  standardize = TRUE
)

pred_ridge <- as.numeric(
  predict(cv_ridge, newx = X_test, s = "lambda.min")
)



# LASSO: alpha = 1

set.seed(123)

cv_lasso <- glmnet::cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 1,
  family = "gaussian",
  nfolds = 10,
  type.measure = "mse",
  standardize = TRUE
)

pred_lasso <- as.numeric(
  predict(cv_lasso, newx = X_test, s = "lambda.min")
)



# ELASTIC-NET: alpha = 0.5

set.seed(123)

cv_elastic <- glmnet::cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 0.5,
  family = "gaussian",
  nfolds = 10,
  type.measure = "mse",
  standardize = TRUE
)

pred_elastic <- as.numeric(
  predict(cv_elastic, newx = X_test, s = "lambda.min")
)



# EVALUACIÓN:

metricas_regularizacion <- data.frame(
  Modelo = c("Ridge", "Lasso", "Elastic-Net"),
  RMSE = c(
    Metrics::rmse(y_test, pred_ridge),
    Metrics::rmse(y_test, pred_lasso),
    Metrics::rmse(y_test, pred_elastic)
  ),
  MAE = c(
    Metrics::mae(y_test, pred_ridge),
    Metrics::mae(y_test, pred_lasso),
    Metrics::mae(y_test, pred_elastic)
  )
)

print(metricas_regularizacion)

# Valores de lambda seleccionados por validación cruzada
cat("Lambda Ridge:", cv_ridge$lambda.min, "\n")
cat("Lambda Lasso:", cv_lasso$lambda.min, "\n")
cat("Lambda Elastic-Net:", cv_elastic$lambda.min, "\n")

# Número de coeficientes distintos de cero en Lasso
coef_lasso <- as.matrix(
  coef(cv_lasso, s = "lambda.min")
)

cat(
  "Coeficientes distintos de cero en Lasso:",
  sum(coef_lasso[-1, 1] != 0),
  "\n"
)



# ==============================================================================================================================
# ==============================================================================================================================



#     7. MODELO GAM

# Para el GAM, se mantendrán int_col y ext_col fuera de la predicción, ya que éstas tienen muchas 
# categorías y no aportan mucha información predictiva(además de hacer mucho mas pesado el análisis para
# el computador)
predictores <- c(
  "brand",
  "model_year",
  "milage",
  "fuel_type",
  "transmission",
  "accident",
  "clean_title"
)

# Aplicar suavizado a las variables numéricas disponibles
variables_suaves <- intersect(
  c("model_year", "milage"),
  predictores
)

# Variables categóricas: se mantienen como efectos paramétricos
variables_factor <- setdiff(
  predictores,
  variables_suaves
)

terminos_gam <- c(
  paste0("s(", variables_suaves, ", k = 10)"),
  variables_factor
)

formula_gam <- as.formula(
  paste("price ~", paste(terminos_gam, collapse = " + "))
)

# Ajustar GAM con los datos de entrenamiento
modelo_gam <- mgcv::gam(
  formula_gam,
  data = train,
  method = "REML",
  na.action = na.exclude
)

summary(modelo_gam)

# Predicciones
pred_gam <- predict(
  modelo_gam,
  newdata = test,
  type = "response"
)

# Métricas
rmse_gam <- Metrics::rmse(test$price, pred_gam)
mae_gam  <- Metrics::mae(test$price, pred_gam)

cat("GAM - RMSE:", rmse_gam, "\n")
cat("GAM - MAE:", mae_gam, "\n")

# Gráficos de los efectos suaves
plot(
  modelo_gam,
  pages = 1,
  shade = TRUE,
  residuals = TRUE
)

# Diagnóstico del GAM
mgcv::gam.check(modelo_gam)



# ==============================================================================================================================
# ==============================================================================================================================



#     7. INFERENCIA: EFECTO DEL AÑO DEL VEHÍCULO

# Coeficiente estimado
coef_anio <- coef(modelo_ols)["model_year"]

# Intervalo de confianza del 95 %
ic_anio <- confint(
  modelo_ols,
  parm = "model_year",
  level = 0.95
)

# Valor p del coeficiente
p_anio <- summary(modelo_ols)$coefficients[
  "model_year", "Pr(>|t|)"
]

# Efecto estimado por cada 10 años de diferencia
efecto_10_anios <- 10 * coef_anio
ic_10_anios <- 10 * ic_anio

# Magnitud relativa respecto de la mediana de precios
precio_mediano <- median(train$price, na.rm = TRUE)

efecto_porcentual <- 100 * efecto_10_anios / precio_mediano

cat("Coeficiente por año:", coef_anio, "\n")
cat("IC 95% del coeficiente:\n")
print(ic_anio)

cat("\nEfecto estimado por 10 años:", efecto_10_anios, "\n")
cat("IC 95% del efecto por 10 años:\n")
print(ic_10_anios)

cat("\nValor p:", p_anio, "\n")
cat(
  "Efecto por 10 años como porcentaje del precio mediano:",
  efecto_porcentual, "%\n"
)

if (p_anio < 0.05) {
  cat("El coeficiente es estadísticamente significativo al 5%.\n")
} else {
  cat("No hay evidencia suficiente de significancia al 5%.\n")
}


# ==============================================================================================================================
# ==============================================================================================================================



#     8. PREPARACIÓN DE LOS DATOS PARA LOS MODELOS

# En este script tenemos un data frame llamado "train" (los datos de
# entrenamiento) y caret tiene una función llamada train(). No se pisan:
# R distingue si usamos "train" como dato o train(...) como función.

# Excluimos las variables "ext_col" e "int_col" porque como éstas tienen demasiados tipos
# el computador se satura y los modelos no funcionan:
vars_excluir <- c("ext_col", "int_col")

# Nos quedamos con todas las columnas MENOS las excluidas (el signo ! significa "no").
train_m <- train[, !(names(train) %in% vars_excluir)]
test_m  <- test[,  !(names(test)  %in% vars_excluir)]

# Convertir texto a "factor"
cols_chr <- names(train_m)[sapply(train_m, is.character)]

# Para cada una hacemos lo siguiente:
for (v in cols_chr) {
  # Lista de las categorías que aparecen en train, sin repetir, más "unknown":
  niveles <- sort(unique(c(train_m[[v]], "unknown")))
  
  # Si en test aparece una categoría que train nunca vio, el modelo no tiene 
  # una columna dummy para ella y se caería con un error de "categoría nueva", en ese caso,
  #la tratamos como "unknown".
  test_m[[v]][!(test_m[[v]] %in% niveles)] <- "unknown"
  
  # NUEVO: "limpiamos" los nombres de las categorías. make.names() reemplaza
  # espacios, guiones, barras y otros símbolos raros por puntos
  # (ej.: "6-Speed A/T" pasa a ser "X6.Speed.A.T"; los nombres que empiezan con
  # un número reciben una "X" delante). 
  
  # Como usamos la MISMA lista (niveles) para train y test, la traducción de
  # nombres es idéntica en ambos y las categorías siguen coincidiendo.
  niveles_limpios <- make.names(niveles, unique = TRUE)
  
  # Convertimos la columna a factor usando esa misma lista en train y en test.
  # levels = nombres originales (para reconocer cada categoría);
  # labels = nombres limpios (los que verá el modelo).
  train_m[[v]] <- factor(train_m[[v]], levels = niveles, labels = niveles_limpios)
  test_m[[v]]  <- factor(test_m[[v]],  levels = niveles, labels = niveles_limpios)
}



# ==============================================================================================================================
# ==============================================================================================================================















# VALIDACIÓN CRUZADA (cómo elegimos la mejor configuración de cada modelo)

# ¿QUÉ ES LA VALIDACIÓN CRUZADA DE 5 FOLDS?
# Imagina que los datos de entrenamiento son un mazo de cartas.
#   1. Lo dividimos al azar en 5 montones iguales (los "folds").
#   2. Entrenamos el modelo con 4 montones y lo "examinamos" con el 5º, que no vio.
#   3. Repetimos 5 veces, rotando cuál montón hace de examen.
#   4. Promediamos las 5 notas. Esa nota promedio dice qué tan bien generaliza
#      el modelo a datos nuevos.
# Así probamos muchas configuraciones (hiperparámetros) y elegimos la mejor,
# SIN tocar el set de test, que se reserva para la evaluación final.

# Creamos los 5 montones UNA sola vez, con semilla fija (set.seed(123) hace que
# el "azar" sea siempre el mismo y los resultados se puedan repetir).
# Usar los mismos montones en todos los modelos es obligatorio para poder
# combinarlos después con caretEnsemble. returnTrain = TRUE devuelve, para cada
# fold, las filas que se usan para ENTRENAR (los otros 4 montones).
# Todos los modelos del proyecto (incluidos los del resto del equipo) deben usar
# este mismo objeto "folds" y la misma hoja "ctrl", para que la comparación
# final de la tabla sea justa.
set.seed(123)
folds <- createFolds(train_m$price, k = 5, returnTrain = TRUE)

# trainControl() es la "hoja de instrucciones" que se le pasa a train():
# ctrl <- trainControl(
#  method          = "cv",        # "cv" = validación cruzada
#  number          = 5,           # 5 folds (5 montones)
#  index           = folds,       # usar exactamente los montones que creamos arriba
#  savePredictions = "final"      # guardar las predicciones de "examen" del mejor modelo;
#  # sin esto, caretEnsemble no puede combinar los modelos
#)

# --- Preprocesamiento SOLO dentro de train() ----------------------------------
# DATA LEAKAGE (fuga de información): es "hacer trampa sin querer", por ejemplo
# calcular el promedio de una variable usando también los datos de examen.
# Para evitarlo, el preprocesamiento se declara en el argumento preProcess de
# train(), y caret lo calcula dentro de cada fold, usando solo sus datos de
# entrenamiento. Qué hace cada opción:
#   "zv"     : (zero variance) elimina columnas que valen siempre lo mismo; no sirven.
#   "center" : resta el promedio a cada variable (queda centrada en 0).
#   "scale"  : divide por la desviación estándar (todas las variables quedan en la misma escala).
#pre <- c("zv", "center", "scale")



# DIAGNÓSTICO DE DIMENSIONALIDAD (cuántas columnas ve realmente el modelo)

# Después de quitar price (la variable a predecir) y las 3 variables excluidas,
# quedan 8 predictores "base". PERO el modelo no ve 8 columnas: ve las columnas
# que resultan del dummy encoding. Una variable con 50 categorías aporta 49
# columnas. Variables como brand, engine y transmission tienen muchas categorías,
# así que el total real es mucho mayor que 8.
# Este dato importa porque Random Forest tiene un parámetro (mtry) que NO puede
# ser mayor que el número real de columnas. Por eso lo calculamos en vez de adivinarlo.

# model.matrix() hace el dummy encoding igual que lo haría caret por dentro.
# El [, -1] elimina la primera columna (el "intercepto", que no es un predictor).
x_dummy <- model.matrix(price ~ ., data = train_m)[, -1, drop = FALSE]

# Número total de columnas después del dummy encoding.
p_total <- ncol(x_dummy)

# Dentro de cada fold, la opción "zv" elimina las columnas que quedan constantes
# (por ejemplo, una categoría que no aparece en las filas de ese fold).
# Entonces, el número de columnas útiles puede ser un poco menor en cada fold.
# Contamos cuántas columnas NO constantes tiene cada fold y nos quedamos con el
# mínimo, ese es el límite seguro para mtry.
p_folds <- vapply(
  folds,
  function(idx) {
    # Para cada columna vemos si tiene al menos un valor distinto del primero
    # (si lo tiene, no es constante). sum() cuenta cuántas columnas cumplen.
    sum(apply(x_dummy[idx, , drop = FALSE], 2, function(col) any(col != col[1])))
  },
  numeric(1)
)
p_min <- min(p_folds)

# Mostramos los números en pantalla para poder citarlos en la presentación.
cat("Predictores base:", ncol(train_m) - 1,
    "| Columnas tras dummy encoding:", p_total,
    "| Mínimo de columnas útiles por fold (tras zv):", p_min, "\n")

# FUNCIÓN PARA MEDIR QUÉ TAN BIEN PREDICE CADA MODELO EN TEST

# Una "función" es una receta reutilizable: le damos las predicciones y los
# precios reales, y nos devuelve las 3 métricas. Así no repetimos código 5 veces.
#
# Las métricas:
#   RMSE : error típico en dólares. Eleva los errores al cuadrado, así que
#          castiga MUCHO los errores grandes (equivocarse por $50.000 pesa mucho).
#   MAE  : error promedio absoluto en dólares ("en promedio me equivoco por $X").
#          Es más fácil de explicar que el RMSE.
#   R²   : porcentaje de la variación del precio que el modelo logra explicar.
#          1 = perfecto, 0 = tan malo como predecir siempre el promedio
#          (puede ser negativo si es peor que eso).
evaluar <- function(pred, obs, nombre) {
  # Aseguramos que las predicciones sean un vector de números simple
  # (los modelos combinados a veces las entregan en otro formato).
  pred <- as.numeric(unlist(pred))
  
  data.frame(
    Modelo = nombre,
    RMSE   = Metrics::rmse(obs, pred),
    MAE    = Metrics::mae(obs, pred),
    R2     = caret::R2(pred, obs, form = "traditional")  # R² clásico: 1 - (error del modelo / error de predecir el promedio)
  )
}










# ==============================================================================================================================
# ==============================================================================================================================



#     9. ÁRBOL DE REGRESIÓN (rpart)

# Se crea la grilla:
grid_rpart <- expand.grid(
  # cp (parámetro de complejidad)
  cp = c(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02)
)

set.seed(123)   # misma "suerte" cada vez que se ejecute
mod_rpart <- train(
  price ~ .,                # fórmula: "predecir price usando todas las demás columnas"
  data       = train_m,     # datos con los que se entrena
  method     = "rpart",     # tipo de modelo: árbol de regresión
  metric     = "RMSE",      # criterio para elegir el mejor cp: el menor RMSE en validación cruzada
  trControl  = ctrl,        # hoja de instrucciones de la validación cruzada (5 folds)
  tuneGrid   = grid_rpart,  # valores de cp que se van a probar
  preProcess = pre          # preprocesamiento hecho DENTRO de cada fold (sin fuga de datos)
)



# ==============================================================================================================================
# ==============================================================================================================================



# 10. RANDOM FOREST (ranger)

# mtry = cuántas columnas "sortea" cada árbol para elegir CADA pregunta.
mtry_cand <- floor(c(sqrt(p_min), p_min / 6, p_min / 3))

# pmax(..., 1) asegura que mtry sea al menos 1; pmin(..., p_min) asegura que
# nunca sea mayor que las columnas disponibles (si lo fuera, ranger da error).
# unique() elimina repetidos y sort() los ordena de menor a mayor.
mtry_cand <- sort(unique(pmin(pmax(mtry_cand, 1), p_min)))

grid_ranger <- expand.grid(
  mtry          = mtry_cand,   # columnas sorteadas en cada pregunta (explicado arriba)
  splitrule     = "variance",  # regla para decidir cada pregunta: elegir la que más reduce la varianza (dispersión) del precio
  min.node.size = c(3, 5, 10)  # mínimo de autos que debe tener un grupo final (hoja):
)

# Seguro de vida: si por algún motivo la grilla tuviera un mtry inválido,
# el programa se detiene aquí, ANTES de gastar tiempo en la validación cruzada.
stopifnot(all(grid_ranger$mtry >= 1), all(grid_ranger$mtry <= p_min))

set.seed(123)
mod_ranger <- train(
  price ~ .,
  data       = train_m,
  method     = "ranger",    # motor de Random Forest
  metric     = "RMSE",
  trControl  = ctrl,
  tuneGrid   = grid_ranger,
  preProcess = pre,
  num.trees  = 300,         # número de árboles del bosque. Más árboles = resultado más estable
  # pero más lento. Es fijo (no se prueba en la grilla).
  importance = "impurity",  # pide calcular qué variables fueron más importantes para el bosque
  verbose    = FALSE        
)



# ==============================================================================================================================
# ==============================================================================================================================



#     11. XGBOOST (método propio compatible con xgboost 3.x)

# Una receta de modelo en caret es una lista con las piezas que train() necesita:
xgb_custom <- list(
  label   = "XGBoost compatible con xgboost 3.x",
  library = "xgboost",          # paquete que debe estar instalado
  type    = "Regression",       # predecimos un número (precio), no una categoría
  
  # Lista de hiperparámetros que este modelo acepta (nombre, tipo y descripción).
  parameters = data.frame(
    parameter = c("nrounds", "max_depth", "eta", "gamma",
                  "colsample_bytree", "min_child_weight", "subsample"),
    class     = rep("numeric", 7),
    label     = c("Rondas de boosting", "Profundidad maxima", "Tasa de aprendizaje",
                  "Ganancia minima", "Fraccion de columnas", "Peso minimo hijo",
                  "Fraccion de filas")
  ),
  
  # Grilla por defecto (no se usa, porque nosotros entregamos tuneGrid propio,
  # pero caret exige que exista).
  grid = function(x, y, len = NULL, search = "grid") {
    expand.grid(nrounds = 100, max_depth = 3, eta = 0.1, gamma = 0,
                colsample_bytree = 0.8, min_child_weight = 1, subsample = 0.8)
  },
  
  loop = NULL,                  # sin atajos: se entrena una vez por cada combinación
  
  # CÓMO SE ENTRENA: caret nos pasa x (columnas ya preprocesadas), y (precios)
  # y param (una fila de la grilla con los valores a probar).
  fit = function(x, y, wts, param, lev, last, classProbs, ...) {
    # xgboost necesita sus datos en un formato propio ("DMatrix"): una matriz numérica + la respuesta.
    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(x), label = y)
    
    # Hiperparámetros agrupados en una lista. Aquí SÍ va el objetivo
    # (que era lo que causaba el aviso de "objective"):
    #   reg:squarederror = regresión minimizando el error al cuadrado.
    #   verbosity = 0    = no imprimir mensajes en pantalla.
    params <- list(
      objective        = "reg:squarederror",
      eta              = param$eta,
      max_depth        = param$max_depth,
      gamma            = param$gamma,
      colsample_bytree = param$colsample_bytree,
      min_child_weight = param$min_child_weight,
      subsample        = param$subsample,
      verbosity        = 0
    )
    
    booster <- xgboost::xgb.train(params = params, data = dtrain,
                                  nrounds = param$nrounds)
    
    # Guardamos el modelo dentro de una lista normal de R, junto con los nombres
    # de las columnas. Así evitamos modificar el objeto de xgboost (el motivo del error).
    list(booster = booster, xNames = colnames(x))
  },
  
  # CÓMO SE PREDICE: ordenamos las columnas igual que en el entrenamiento y
  # le pedimos al modelo el precio estimado de cada auto.
  predict = function(modelFit, newdata, submodels = NULL) {
    newdata <- as.matrix(newdata)[, modelFit$xNames, drop = FALSE]
    predict(modelFit$booster, newdata)
  },
  
  prob = NULL,                  # solo se usa en clasificación
  sort = function(x) x[order(x$nrounds, x$max_depth, x$eta), ]   # orden de la tabla de resultados
)

# La grilla de hiperparámetros NO cambia respecto a la versión anterior.
grid_xgb <- expand.grid(
  nrounds          = c(100, 300),  # número de árboles en la fila (cuántas rondas de corrección).
  # Pocas rondas = modelo flojo; demasiadas = memoriza los datos.
  max_depth        = c(3, 6),      # profundidad máxima de cada árbol (cuántas preguntas seguidas puede hacer).
  # 3 = árboles simples; 6 = árboles más complejos.
  eta              = c(0.05, 0.1), # "shrinkage" (tasa de aprendizaje): cuánto le creemos a cada árbol nuevo.
  # Valor chico = correcciones pequeñas y cuidadosas (más seguro pero necesita
  # más árboles); valor grande = correcciones fuertes (más rápido pero arriesgado).
  gamma            = 0,            # ganancia mínima exigida para hacer una pregunta nueva en un árbol.
  # 0 = no se exige nada extra (el árbol puede crecer libremente).
  colsample_bytree = 0.8,          # fracción de columnas que usa cada árbol: 0.8 = 80%. Es una PROPORCIÓN,
  # no una cantidad fija, así que nunca supera las columnas disponibles.
  min_child_weight = 1,            # tamaño mínimo (en "peso") que debe tener un grupo para poder dividirse.
  # 1 = permisivo; valores altos = más conservador.
  subsample        = 0.8           # fracción de filas que usa cada árbol: 0.8 = 80% de los autos, elegidos al azar.
  # Agrega variedad y reduce el riesgo de memorizar.
)

set.seed(123)
mod_xgb <- train(
  price ~ .,
  data       = train_m,
  method     = xgb_custom,  # en vez de "xgbTree": usamos la receta propia definida arriba
  metric     = "RMSE",
  trControl  = ctrl,
  tuneGrid   = grid_xgb,
  preProcess = pre
)



# ==============================================================================================================================
# ==============================================================================================================================




#     12. ENSAMBLE (AVERAGING Y STACKING)

# Se juntan 3 modelos ya entrenados en una "lista de expertos".
# (Esto es posible porque todos usaron los mismos folds y savePredictions = "final".)
lista_modelos <- as.caretList(
  list(rpart = mod_rpart, ranger = mod_ranger, xgb = mod_xgb)
)

# Hoja de instrucciones para el "meta-modelo", el que decide cómo mezclar a los expertos.
# NUEVO: también usa los mismos 5 folds (index = folds) que los modelos base.
ctrl_meta <- trainControl(
  method          = "cv",
  number          = 5,
  index           = folds,
  savePredictions = "final"
)

# a. AVERAGING PONDERADO 

set.seed(123)
ens_avg <- caretEnsemble(
  lista_modelos,
  metric    = "RMSE",       # elegir los pesos que minimicen el RMSE
  trControl = ctrl_meta
)

# b.STACKING (caretStack) 
set.seed(123)
ens_stack <- caretStack(
  lista_modelos,
  method    = "glm",        # tipo de meta-modelo (regresión lineal)
  metric    = "RMSE",
  trControl = ctrl_meta
)



# ==============================================================================================================================
# ==============================================================================================================================



# EVALUACIÓN FINAL EN EL SET DE TEST

# predict() le pide a cada modelo que estime el precio de esos autos; caret aplica solo el mismo
# preprocesamiento aprendido en entrenamiento.
pred_rpart  <- predict(mod_rpart,  newdata = test_m)
pred_ranger <- predict(mod_ranger, newdata = test_m)
pred_xgb    <- predict(mod_xgb,    newdata = test_m)
pred_avg    <- predict(ens_avg,    newdata = test_m)
pred_stack  <- predict(ens_stack,  newdata = test_m)

# Se calculan las métricas de cada modelo con ña función y se apilan
# una debajo de otra (rbind) en una sola tabla.
resultados <- rbind(
  evaluar(pred_rpart,  test_m$price, "Árbol de regresión (rpart)"),
  evaluar(pred_ranger, test_m$price, "Random Forest (ranger)"),
  evaluar(pred_xgb,    test_m$price, "XGBoost (xgbTree)"),
  evaluar(pred_avg,    test_m$price, "Ensamble Averaging (caretEnsemble)"),
  evaluar(pred_stack,  test_m$price, "Ensamble Stacking (caretStack, glm)")
)

# Se ordena la tabla del mejor al peor (menor RMSE arriba) y limpiamos la numeración.
resultados <- resultados[order(resultados$RMSE), ]
rownames(resultados) <- NULL

# Tabla final:
print(resultados)

# Hiperparámetros ganadores de cada modelo (la configuración que obtuvo el mejor
# RMSE en la validación cruzada). Sirven para justificar las decisiones en la presentación.
mod_rpart$bestTune
mod_ranger$bestTune
mod_xgb$bestTune