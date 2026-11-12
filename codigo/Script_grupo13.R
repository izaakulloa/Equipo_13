
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
# Buscar el CSV tanto en la carpeta del proyecto como en la estructura del grupo.
# No se cambian los datos ni la particion segun donde este el archivo.
rutas_posibles <- c("used_cars.csv", "../datos/used_cars.csv", "datos/used_cars.csv")
rutas_encontradas <- rutas_posibles[file.exists(rutas_posibles)]
if (length(rutas_encontradas) == 0L) {
  stop("Falta used_cars.csv: ubicarlo junto al script o en ../datos/.")
}
ruta <- rutas_encontradas[1]
cat("Base de datos:", ruta, "\n")

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

# Duplicados: se revisan ANTES de partir (si hubiera filas repetidas, se eliminarían aquí,
# para que una misma fila no pueda caer a la vez en train y en test).
sum(duplicated(datos)) # No hay duplicados

set.seed(123)  # Reproducibilidad

n <- nrow(datos)
indices_train <- sample(1:n, size = 0.8 * n)

train <- datos[indices_train, ]
test  <- datos[-indices_train, ]
stopifnot(nrow(train) + nrow(test) == nrow(datos),
          length(intersect(rownames(train), rownames(test))) == 0L)
cat("Division train/test:", nrow(train), "/", nrow(test), "filas.\n")



# ==============================================================================================================================
# ==============================================================================================================================



#     3. LIMPIEZA DE DATOS:

# Las reglas de limpieza son FIJAS: no se aprenden de los datos (convertir símbolos en NA,
# reemplazar NA por la etiqueta "unknown", quitar "$" y "mi."). Se definen una sola vez
# en una función y se aplican por separado a train y a test.
# La exploración (conteos, rangos, resúmenes) se hace SOLO sobre train.

# Se revisan los primeros detos del conjunto:
head(train)
str(train)
dim(train)

# Valores faltantes:
colSums(is.na(train))    # No hay valores NA en este dataset

# Valores "unknown" por columna:
sapply(train, function(x) sum(x == "unknown")) # No hay valores "unknown"

# Contar cadenas vacías por columna
sapply(train, function(x) {
  if (is.character(x) || is.factor(x)) {
    sum(trimws(as.character(x)) == "", na.rm = TRUE)
  } else {
    0
  }
})

# Encontrar el tipo de "guion" presente en las casillas, ya que no es el típico:
pos_guion <- which(train$ext_col == "–")[1]
if (!is.na(pos_guion)) {
  print(train$ext_col[pos_guion])
  print(charToRaw(train$ext_col[pos_guion]))
}

# Contar cadenas vacías por columna
sapply(train, function(x) {
  if (is.character(x) || is.factor(x)) {
    sum(trimws(as.character(x)) == "–", na.rm = TRUE)
  } else {
    0
  }
})

# Copia temporal de train, solo para mirar cuántos NA aparecen al convertir
# las cadenas vacías y los guiones (no se usa después):
train_na <- train
train_na[train_na == ""] <- NA
train_na[train_na == "–"] <- NA

# Cantidad de NA por columna
colSums(is.na(train_na))
# Porcentaje de NA por columna
round(colMeans(is.na(train_na)) * 100, 2)
rm(train_na)

# Reglas de limpieza (idénticas para train y test):
limpiar_datos <- function(df) {

  df[df == ""] <- NA
  df[df == "–"] <- NA

  # Se intercambian los NA de la variable NA por "unknown" ya que si se eliminaran las filas con datos nulos,
  # se perdería una cantidad considerable de información:
  df$clean_title[is.na(df$clean_title)] <- "unknown"
  df$accident[is.na(df$accident)] <- "unknown"
  df$fuel_type[is.na(df$fuel_type)] <- "unknown"
  df$engine[is.na(df$engine)] <- "unknown"
  df$transmission[is.na(df$transmission)] <- "unknown"
  df$ext_col[is.na(df$ext_col)] <- "unknown"
  df$int_col[is.na(df$int_col)] <- "unknown"

  # Traspasar datos numéricos a clase numeric:

  # Para price y milage, hay que limpiar la variable para que no imprima error por los símbolos:
  df$price <- as.numeric(
    gsub("[$,]", "", trimws(as.character(df$price)))
  )

  df$milage <- as.numeric(
    gsub(",", "",
         gsub(" mi\\.", "", as.character(df$milage)))
  )

  # En el caso del año del modelo, se hace conversión directa:
  df$model_year <- as.numeric(as.character(df$model_year))

  df
}

train <- limpiar_datos(train)
test  <- limpiar_datos(test)

# Valores "unknown" por columna:
sapply(train, function(x) sum(x == "unknown"))

# Duplicados:
sum(duplicated(train)) # No hay duplicados

# Rangos razonables:
range(train$model_year) # 1974 hasta 2024
range(train$milage)  # 1,000 mi hasta 99,999 mi
range(train$price)   # $1,599,000 hasta $99,999


# No se eliminan variables, todas sirven


summary(train)
names(train)




# ==============================================================================================================================
# ==============================================================================================================================



#     4. BASELINES: DEFINICION SIN USAR EL TEST

# Aqui solo se definen estadisticos de entrenamiento.
# Los errores de los baselines se calculan mas adelante con los 5 folds.
# La muestra de test se mantendra reservada hasta elegir el modelo final.
media_global <- mean(train$price, na.rm = TRUE)
promedio_por_marca <- tapply(train$price, as.character(train$brand), mean)
cat("Baselines preparados con datos de entrenamiento.\n")


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

# Se selecciona y compara el modelo por validacion cruzada mas adelante.
# No se consulta el precio del test aqui.

# Intervalos de confianza de los coeficientes
confint(modelo_ols, level = 0.95)


# DIAGNNÓSTICO DE RESIDUOS

residuos <- residuals(modelo_ols)
ajustados <- fitted(modelo_ols)

# Guardar graficos, sin depender del tamano de la ventana Plots.
# Esto no modifica los modelos ni sus estimaciones.
pdf("diagnostico_OLS.pdf", width = 11, height = 9)
par(mfrow = c(2, 2))
try(plot(modelo_ols), silent = TRUE)
dev.off()

# Heterocedasticidad: test de Breusch-Pagan
test_bp <- lmtest::bptest(modelo_ols)
print(test_bp)

# Grafico Q-Q guardado en PDF.
pdf("qq_residuos_OLS.pdf", width = 8, height = 6)
qqnorm(residuos, main = "Q-Q plot de residuos OLS")
qqline(residuos, col = "red")
dev.off()

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
# Cuando hay colinealidad exacta, car::vif puede fallar.
# Se informa el problema sin detener la evaluacion predictiva.
vif_ols <- tryCatch(car::vif(modelo_ols),
                    error = function(e) paste("VIF no disponible:", conditionMessage(e)))
print(vif_ols)



# ==============================================================================================================================
# ==============================================================================================================================



#     6. MATRICES DE DISEÑO PARA GLMNET

# Crear matrices a partir de los mismos predictores
X_train <- model.matrix(
  formula_ols,
  data = train
)[, -1, drop = FALSE]

# La matriz del test se crea SOLAMENTE al terminar la seleccion por CV.
y_train <- train$price

# 5 FOLDS COMPARTIDOS PARA TODO EL PROYECTO:
# se crean UNA sola vez, con set.seed(123) y SOLO con train. Los usan glmnet (aquí),
# rpart, ranger, xgboost y los ensambles (más abajo), para que la comparación sea justa.
set.seed(123)
folds <- createFolds(train$price, k = 5, returnTrain = TRUE)

# glmnet no recibe la lista de filas de entrenamiento, sino un vector que dice a qué
# fold pertenece cada fila (1 a 5). Se arma con los mismos folds: las filas que NO están
# en el entrenamiento del fold k son las de examen del fold k.
foldid <- integer(nrow(train))
for (k in seq_along(folds)) foldid[-folds[[k]]] <- k
stopifnot(all(foldid > 0), nrow(X_train) == length(foldid))



# RIDGE: alpha = 0

set.seed(123)

cv_ridge <- glmnet::cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 0,
  family = "gaussian",
  foldid = foldid,     # mismos 5 folds que el resto de los modelos (antes: nfolds = 10)
  type.measure = "mse",
  standardize = TRUE
)

# La prediccion de ridge en test se hara solo si gana en CV.



# LASSO: alpha = 1

set.seed(123)

cv_lasso <- glmnet::cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 1,
  family = "gaussian",
  foldid = foldid,     # mismos 5 folds que el resto de los modelos (antes: nfolds = 10)
  type.measure = "mse",
  standardize = TRUE
)

# La prediccion de lasso en test se hara solo si gana en CV.



# ELASTIC-NET: alpha = 0.5

set.seed(123)

cv_elastic <- glmnet::cv.glmnet(
  x = X_train,
  y = y_train,
  alpha = 0.5,
  family = "gaussian",
  foldid = foldid,     # mismos 5 folds que el resto de los modelos (antes: nfolds = 10)
  type.measure = "mse",
  standardize = TRUE
)

# La prediccion de elastic en test se hara solo si gana en CV.



# Los errores de Ridge, Lasso y Elastic Net se compararan
# con validacion cruzada sobre train; no con test.

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

# El GAM tambien se comparara mediante validacion cruzada sobre train.

# Se guardan los graficos del GAM en un PDF para que no fallen en tablet.
pdf("diagnostico_GAM.pdf", width = 11, height = 9)
try(plot(modelo_gam, pages = 1, shade = TRUE, residuals = TRUE), silent = TRUE)
# gam.check tambien intenta abrir un grafico; dentro del PDF no hay limite
# de tamano de ventana como en Posit Cloud.
try(mgcv::gam.check(modelo_gam), silent = TRUE)
dev.off()


# ==============================================================================================================================
# ==============================================================================================================================

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
# Solo se preparan las variables explicativas del test; no se consultan
# ni se utilizan sus precios hasta despues de la seleccion final.

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
# El objeto "folds" ya se creó en la sección 6 (justo antes de Ridge/Lasso/Elastic-Net),
# con set.seed(123) y usando SOLO train, para que TODOS los modelos usen los mismos 5 montones.
stopifnot(length(folds) == 5)

# trainControl() es la "hoja de instrucciones" que se le pasa a train():
ctrl <- trainControl(
  method          = "cv",        # "cv" = validación cruzada
  number          = 5,           # 5 folds (5 montones)
  index           = folds,       # usar exactamente los montones que creamos arriba
  savePredictions = "final"      # guardar las predicciones de "examen" del mejor modelo;
  # sin esto, caretEnsemble no puede combinar los modelos
)

# Preprocesamiento SOLO dentro de train() 
pre <- c("zv", "center", "scale")



# DIAGNÓSTICO DE DIMENSIONALIDAD (cuántas columnas ve realmente el modelo)

# Después de quitar price (la variable a predecir) y las 2 variables excluidas,
# quedan predictores "base". PERO el modelo no ve 8 columnas: ve las columnas
# que resultan del dummy encoding, una variable con 50 categorías aporta 49
# columnas. Variables como brand, engine y transmission tienen muchas categorías,
# así que el total real es mucho mayor que 8.
# Este dato importa porque Random Forest tiene un parámetro (mtry) que no puede
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



# 13. VALIDACION CRUZADA Y TABLA COMPARATIVA

# Usa los 5 folds, los modelos y las configuraciones ya definidos.
# No vuelve a crear los modelos definitivos: ajusta copias por fold
# cuando es indispensable para estimar el error fuera del entrenamiento.
# Averaging y Stacking se validan con folds externos; esto puede tardar.
# La eleccion por CV que se imprime abajo no debe cambiarse mirando test.

# Verificar que cada auto aparezca en un unico fold de validacion.
objetos_necesarios <- c("train", "folds", "foldid", "X_train", "y_train",
                       "formula_ols", "mod_rpart", "mod_ranger", "mod_xgb")
faltantes <- objetos_necesarios[!vapply(objetos_necesarios, exists, logical(1))]
if (length(faltantes) > 0) {
  stop("Antes de esta seccion deben existir: ", paste(faltantes, collapse = ", "))
}

n_cv <- nrow(train)
stopifnot(length(folds) == 5, length(foldid) == n_cv,
          nrow(X_train) == n_cv, length(y_train) == n_cv)
indices_validacion <- lapply(folds, function(ind_entrenamiento) {
  setdiff(seq_len(n_cv), ind_entrenamiento)
})
stopifnot(identical(sort(as.integer(unlist(indices_validacion))), seq_len(n_cv)))
stopifnot(all(is.finite(train$price)))

# Calcular RMSE, MAE y R2 con predicciones fuera de entrenamiento.
# Tambien se mide el RMSE de los autos de 2020 en adelante y la variacion
# del RMSE entre los cinco folds.
resumir_validacion <- function(nombre, predicciones, grupos = foldid) {
  real <- train$price
  if (length(predicciones) != length(real) || any(!is.finite(predicciones))) {
    stop("Predicciones incompletas o invalidas en: ", nombre)
  }
  errores <- real - predicciones
  rmse <- sqrt(mean(errores^2))
  mae <- mean(abs(errores))
  r2 <- 1 - sum(errores^2) / sum((real - mean(real))^2)
  rmse_por_fold <- vapply(seq_len(5), function(k) {
    idx <- which(grupos == k)
    sqrt(mean(errores[idx]^2))
  }, numeric(1))
  autos_recientes <- is.finite(train$model_year) & train$model_year >= 2020
  rmse_recientes <- if (any(autos_recientes)) {
    sqrt(mean(errores[autos_recientes]^2))
  } else NA_real_
  data.frame(
    Modelo = nombre,
    RMSE = rmse,
    MAE = mae,
    R2 = r2,
    RMSE_2020_mas = rmse_recientes,
    Error_estandar_RMSE = sd(rmse_por_fold) / sqrt(5),
    Estado = "Calculado",
    stringsAsFactors = FALSE
  )
}

# 3. Baselines: en cada fold, calcular promedios SOLO con sus filas
# de entrenamiento; despues predecir las filas reservadas de ese fold.
pred_media_cv <- pred_marca_cv <- pred_previo_cv <- rep(NA_real_, n_cv)
ids_originales <- as.integer(rownames(train))

for (k in seq_len(5)) {
  idx_ent <- folds[[k]]
  idx_val <- indices_validacion[[k]]
  promedio <- mean(train$price[idx_ent])

  pred_media_cv[idx_val] <- promedio

  medias_marca <- tapply(train$price[idx_ent],
                        as.character(train$brand[idx_ent]), mean)
  estimadas_marca <- unname(medias_marca[as.character(train$brand[idx_val])])
  estimadas_marca[is.na(estimadas_marca)] <- promedio
  pred_marca_cv[idx_val] <- estimadas_marca

  # Referencia de "valor previo": se basa en el orden de filas, NO en fechas.
  # Es un baseline ilustrativo, porque el archivo no tiene orden temporal real.
  ids_ordenados <- idx_ent[order(ids_originales[idx_ent])]
  posicion <- findInterval(ids_originales[idx_val], ids_originales[ids_ordenados])
  pred_previo_cv[idx_val] <- ifelse(
    posicion == 0, promedio,
    train$price[ids_ordenados[pmax(posicion, 1)]]
  )
}

tabla_validacion <- rbind(
  resumir_validacion("Baseline: promedio general", pred_media_cv),
  resumir_validacion("Baseline: promedio por marca", pred_marca_cv),
  resumir_validacion("Baseline: valor previo (sin fechas)", pred_previo_cv)
)

# Preparar factores de cada fold usando SOLO sus filas de entrenamiento.
# Si en el fold de validacion aparece una categoria no vista durante el ajuste,
# se reemplaza por la categoria mas frecuente del fold de entrenamiento.
preparar_factores_fold <- function(df_ent, df_val, variables) {
  for (v in intersect(variables, names(df_ent))) {
    if (is.character(df_ent[[v]]) || is.factor(df_ent[[v]])) {
      a <- as.character(df_ent[[v]])
      b <- as.character(df_val[[v]])
      disponibles <- unique(a[!is.na(a)])
      if (length(disponibles) == 0) stop("Factor sin valores: ", v)
      frecuente <- names(which.max(table(a)))
      b[is.na(b) | !(b %in% disponibles)] <- frecuente
      df_ent[[v]] <- factor(a, levels = disponibles)
      df_val[[v]] <- factor(b, levels = disponibles)
    }
  }
  list(ent = df_ent, val = df_val)
}

# Regresion lineal y GAM: volver a ajustar cada modelo en cuatro folds
# y predecir el fold que quedo fuera. Esto NO usa el test final.
pred_ols_cv <- pred_gam_cv <- rep(NA_real_, n_cv)
variables_ols <- all.vars(formula_ols)[-1]
variables_gam <- c("brand", "model_year", "milage", "fuel_type",
                   "transmission", "accident", "clean_title")

for (k in seq_len(5)) {
  idx_ent <- folds[[k]]
  idx_val <- indices_validacion[[k]]
  d <- preparar_factores_fold(
    train[idx_ent, , drop = FALSE],
    train[idx_val, , drop = FALSE],
    unique(c(variables_ols, variables_gam))
  )

  # Omitir un factor solo si tiene una unica categoria en este fold.
  pred_ols_fold <- variables_ols[vapply(variables_ols, function(v) {
    !is.factor(d$ent[[v]]) || nlevels(d$ent[[v]]) > 1
  }, logical(1))]
  modelo_ols_fold <- lm(reformulate(pred_ols_fold, response = "price"), data = d$ent)
  pred_ols_cv[idx_val] <- as.numeric(predict(modelo_ols_fold, newdata = d$val))

  # GAM conserva los terminos suaves del codigo grupal.
  suaves <- c("model_year", "milage")
  factores <- setdiff(variables_gam, suaves)
  factores <- factores[vapply(factores, function(v) {
    !is.factor(d$ent[[v]]) || nlevels(d$ent[[v]]) > 1
  }, logical(1))]
  terminos_gam_fold <- c(paste0("s(", suaves, ", k = 10)"), factores)
  formula_gam_fold <- as.formula(paste("price ~", paste(terminos_gam_fold,
                                                      collapse = " + ")))
  modelo_gam_fold <- mgcv::gam(formula_gam_fold, data = d$ent, method = "REML")
  pred_gam_cv[idx_val] <- as.numeric(predict(modelo_gam_fold,
                                            newdata = d$val, type = "response"))
}

tabla_validacion <- rbind(
  tabla_validacion,
  resumir_validacion("Regresion lineal (OLS)", pred_ols_cv),
  resumir_validacion("GAM", pred_gam_cv)
)

# Ridge, Lasso y Elastic Net: para evaluar un fold externo,
# elegir lambda mediante una validacion INTERNA usando solo sus cuatro folds
# de entrenamiento. Esto evita elegir lambda usando los precios del fold externo.
prediccion_regularizada_cv <- function(alpha_modelo) {
  pred <- rep(NA_real_, n_cv)
  for (k in seq_len(5)) {
    idx_ent <- folds[[k]]
    idx_val <- indices_validacion[[k]]
    set.seed(123 + k)
    ajuste_interno <- glmnet::cv.glmnet(
      x = X_train[idx_ent, , drop = FALSE],
      y = y_train[idx_ent],
      alpha = alpha_modelo,
      family = "gaussian",
      nfolds = 5,
      type.measure = "mse",
      standardize = TRUE
    )
    pred[idx_val] <- as.numeric(predict(
      ajuste_interno, newx = X_train[idx_val, , drop = FALSE], s = "lambda.min"
    ))
  }
  pred
}

tabla_validacion <- rbind(
  tabla_validacion,
  resumir_validacion("Ridge", prediccion_regularizada_cv(0)),
  resumir_validacion("Lasso", prediccion_regularizada_cv(1)),
  resumir_validacion("Elastic Net", prediccion_regularizada_cv(0.5))
)

# Recuperar las predicciones guardadas por caret durante los cinco folds.
# Se conservan solo las predicciones de la configuracion elegida por CV.
# No hay que volver a entrenar arbol, Random Forest o XGBoost.
predicciones_caret_cv <- function(modelo, nombre) {
  p <- modelo$pred
  if (is.null(p) || nrow(p) == 0) {
    stop("No se guardaron predicciones CV para ", nombre)
  }
  if (!is.null(modelo$bestTune)) {
    for (v in names(modelo$bestTune)) {
      if (!v %in% names(p)) next
      valor <- modelo$bestTune[[v]][1]
      filtro <- if (is.numeric(p[[v]])) {
        abs(p[[v]] - valor) < 1e-8
      } else {
        as.character(p[[v]]) == as.character(valor)
      }
      p <- p[!is.na(filtro) & filtro, , drop = FALSE]
    }
  }
  if (!identical(sort(as.integer(p$rowIndex)), seq_len(n_cv))) {
    stop("Las predicciones de ", nombre,
         " no cubren exactamente una vez cada fila de train")
  }
  pred <- rep(NA_real_, n_cv)
  pred[as.integer(p$rowIndex)] <- as.numeric(p$pred)
  pred
}

tabla_validacion <- rbind(
  tabla_validacion,
  resumir_validacion("Arbol de regresion", predicciones_caret_cv(mod_rpart, "Arbol")),
  resumir_validacion("Random Forest", predicciones_caret_cv(mod_ranger, "Random Forest")),
  resumir_validacion("XGBoost", predicciones_caret_cv(mod_xgb, "XGBoost"))
)



# Guardar un respaldo de la comparacion antes de validar los ensambles.
# Si la validacion de ensambles consume demasiados recursos, los modelos
# individuales siguen teniendo sus resultados de CV disponibles.
write.csv(tabla_validacion, "tabla_cv_modelos_individuales.csv", row.names = FALSE)
cat("Respaldo de modelos individuales: tabla_cv_modelos_individuales.csv\n")

# VALIDACION EXTERNA DE AVERAGING Y STACKING

# En el ensamble se aprenden los pesos de combinacion utilizando predicciones
# de otros modelos. Por eso no basta con medir su error en las predicciones
# que se usaron para aprender esos pesos (daria una estimacion optimista).

# Alinear niveles de factores para que la validacion externa no use
# categorias de las filas reservadas al preparar el modelo.
preparar_datos_ensemble_fold <- function(ent, val) {
  ent <- as.data.frame(ent)
  val <- as.data.frame(val)
  for (v in setdiff(names(ent), "price")) {
    if (is.factor(ent[[v]]) || is.character(ent[[v]])) {
      x <- as.character(ent[[v]])
      y <- as.character(val[[v]])
      x[is.na(x)] <- "unknown"
      y[is.na(y)] <- "unknown"
      niveles <- unique(x)
      if (length(niveles) < 2) {
        stop("El predictor categórico ", v, " tiene un solo nivel en este fold")
      }
      frecuente <- names(which.max(table(x)))
      y[!(y %in% niveles)] <- frecuente
      etiquetas <- make.names(niveles, unique = TRUE)
      ent[[v]] <- factor(x, levels = niveles, labels = etiquetas)
      val[[v]] <- factor(y, levels = niveles, labels = etiquetas)
    }
  }
  list(ent = ent, val = val)
}

pred_avg_cv <- pred_stack_cv <- rep(NA_real_, n_cv)

for (k in seq_along(folds)) {
  cat("Validando ensambles: fold externo", k, "de", length(folds), "\n")
  idx_ent <- folds[[k]]
  idx_val <- indices_validacion[[k]]
  d <- preparar_datos_ensemble_fold(
    train_m[idx_ent, , drop = FALSE],
    train_m[idx_val, , drop = FALSE]
  )
  set.seed(900 + k)
  folds_internos <- caret::createFolds(d$ent$price, k = 3, returnTrain = TRUE)
  control_interno <- caret::trainControl(
    method = "cv", number = 3, index = folds_internos,
    savePredictions = "final", allowParallel = FALSE
  )

  # La grilla y los parametros son los mismos que en el codigo principal.
  # En cada fold externo los parametros se eligen nuevamente: no se copian
  # los hiperparametros elegidos con la base completa.
  arbol_k <- caret::train(
    price ~ ., data = d$ent, method = "rpart", metric = "RMSE",
    trControl = control_interno, tuneGrid = grid_rpart,
    preProcess = pre
  )

  # mtry nunca debe superar el numero de columnas de este fold externo.
  x_k <- model.matrix(price ~ ., data = d$ent)[, -1, drop = FALSE]
  n_columnas_k <- min(vapply(folds_internos, function(ii) {
    sum(apply(x_k[ii, , drop = FALSE], 2, function(z) any(z != z[1])))
  }, numeric(1)))
  grid_rf_k <- grid_ranger[grid_ranger$mtry <= n_columnas_k, , drop = FALSE]
  if (nrow(grid_rf_k) == 0) {
    grid_rf_k <- expand.grid(mtry = 1, splitrule = "variance", min.node.size = 5)
  }
  set.seed(1000 + k)
  rf_k <- caret::train(
    price ~ ., data = d$ent, method = "ranger", metric = "RMSE",
    trControl = control_interno, tuneGrid = grid_rf_k,
    preProcess = pre, num.trees = 300, importance = "impurity", verbose = FALSE
  )

  set.seed(1100 + k)
  xgb_k <- caret::train(
    price ~ ., data = d$ent, method = xgb_custom, metric = "RMSE",
    trControl = control_interno, tuneGrid = grid_xgb,
    preProcess = pre
  )

  # Aprender los pesos y el metamodelo SOLO con las filas de este fold externo.
  lista_k <- caretEnsemble::as.caretList(
    list(rpart = arbol_k, ranger = rf_k, xgb = xgb_k)
  )
  control_meta_k <- caret::trainControl(
    method = "cv", number = 3, index = folds_internos,
    savePredictions = "final"
  )
  set.seed(1200 + k)
  avg_k <- caretEnsemble::caretEnsemble(
    lista_k, metric = "RMSE", trControl = control_meta_k
  )
  set.seed(1300 + k)
  stack_k <- caretEnsemble::caretStack(
    lista_k, method = "glm", metric = "RMSE", trControl = control_meta_k
  )

  a <- as.numeric(unlist(predict(avg_k, newdata = d$val)))
  s <- as.numeric(unlist(predict(stack_k, newdata = d$val)))
  stopifnot(length(a) == length(idx_val), length(s) == length(idx_val))
  pred_avg_cv[idx_val] <- a
  pred_stack_cv[idx_val] <- s
  rm(arbol_k, rf_k, xgb_k, lista_k, avg_k, stack_k, x_k)
  invisible(gc(verbose = FALSE))
}

tabla_validacion <- rbind(
  tabla_validacion,
  resumir_validacion("Averaging", pred_avg_cv),
  resumir_validacion("Stacking", pred_stack_cv)
)

# 9. Comparar frente a un baseline calculado sin usar test.
rmse_baseline <- tabla_validacion$RMSE[
  tabla_validacion$Modelo == "Baseline: promedio por marca"
]
tabla_validacion$Mejora_vs_baseline_pct <-
  100 * (rmse_baseline - tabla_validacion$RMSE) / rmse_baseline

# El baseline de "valor previo" es solo ilustrativo: el CSV no contiene fechas.
# No se considera en la eleccion del modelo final.
tabla_validacion$Elegible <- tabla_validacion$Modelo != "Baseline: valor previo (sin fechas)"
tabla_validacion <- tabla_validacion[order(tabla_validacion$RMSE), , drop = FALSE]
rownames(tabla_validacion) <- NULL

# Se guarda la tabla completa para la presentacion.
tabla_final_cv <- tabla_validacion
tabla_final_cv$RMSE <- round(tabla_final_cv$RMSE, 2)
tabla_final_cv$MAE <- round(tabla_final_cv$MAE, 2)
tabla_final_cv$R2 <- round(tabla_final_cv$R2, 4)
tabla_final_cv$RMSE_2020_mas <- round(tabla_final_cv$RMSE_2020_mas, 2)
tabla_final_cv$Error_estandar_RMSE <- round(tabla_final_cv$Error_estandar_RMSE, 2)
tabla_final_cv$Mejora_vs_baseline_pct <- round(tabla_final_cv$Mejora_vs_baseline_pct, 2)
write.csv(tabla_final_cv, "tabla_comparativa_validacion.csv", row.names = FALSE)
print(tabla_final_cv)

# Se elige el mejor modelo SOLO segun el RMSE de validacion.
candidatos <- tabla_validacion[tabla_validacion$Elegible, , drop = FALSE]
modelo_elegido <- candidatos$Modelo[which.min(candidatos$RMSE)]
cat("Modelo elegido por validacion cruzada:", modelo_elegido, "\n")


# La clasificacion anterior se decide EXCLUSIVAMENTE con train.
# El test se consulta una sola vez en el bloque siguiente.


# Evaluación final única:

# No se ordenan modelos por sus errores de test. El resultado de test NO
# puede utilizarse para cambiar la eleccion hecha con la tabla de CV.

# Se utiliza la misma codificacion del entrenamiento para la matriz de test.
X_test <- model.matrix(formula_ols, data = test)[, -1, drop = FALSE]
X_test <- X_test[, colnames(X_train), drop = FALSE]

# Prediccion del baseline por marca para comparar con el sistema final.
pred_baseline_test <- unname(promedio_por_marca[as.character(test$brand)])
pred_baseline_test[is.na(pred_baseline_test)] <- media_global

# La prediccion final se calcula UNICAMENTE para el ganador de la CV.
predecir_ganador <- function(nombre) {
  if (nombre == "Baseline: promedio general") {
    return(rep(media_global, nrow(test)))
  }
  if (nombre == "Baseline: promedio por marca") {
    return(pred_baseline_test)
  }
  if (nombre == "Regresion lineal (OLS)") {
    return(as.numeric(predict(modelo_ols, newdata = test)))
  }
  if (nombre == "GAM") {
    return(as.numeric(predict(modelo_gam, newdata = test, type = "response")))
  }
  if (nombre == "Ridge") {
    return(as.numeric(predict(cv_ridge, newx = X_test, s = "lambda.min")))
  }
  if (nombre == "Lasso") {
    return(as.numeric(predict(cv_lasso, newx = X_test, s = "lambda.min")))
  }
  if (nombre == "Elastic Net") {
    return(as.numeric(predict(cv_elastic, newx = X_test, s = "lambda.min")))
  }
  if (nombre == "Arbol de regresion") {
    return(as.numeric(predict(mod_rpart, newdata = test_m)))
  }
  if (nombre == "Random Forest") {
    return(as.numeric(predict(mod_ranger, newdata = test_m)))
  }
  if (nombre == "XGBoost") {
    return(as.numeric(predict(mod_xgb, newdata = test_m)))
  }
  if (nombre == "Averaging") {
    return(as.numeric(unlist(predict(ens_avg, newdata = test_m))))
  }
  if (nombre == "Stacking") {
    return(as.numeric(unlist(predict(ens_stack, newdata = test_m))))
  }
  stop("Modelo elegido sin funcion de prediccion: ", nombre)
}

pred_final <- predecir_ganador(modelo_elegido)
precios_test <- test$price  # Primera lectura de los valores reales de test.
stopifnot(length(pred_final) == length(precios_test),
          length(pred_baseline_test) == length(precios_test),
          all(is.finite(pred_final)), all(is.finite(precios_test)))

# Se reporta el resultado final junto al baseline, pero NO se cambia
# la seleccion segun cual tenga mejor resultado de test.
metricas_test <- function(nombre, pred) {
  errores <- precios_test - pred
  ss_total <- sum((precios_test - mean(precios_test))^2)
  nuevos <- test$model_year >= 2020 & is.finite(test$model_year)
  data.frame(
    Modelo = nombre,
    N_test = length(precios_test),
    RMSE = sqrt(mean(errores^2)),
    MAE = mean(abs(errores)),
    R2 = if (ss_total > 0) 1 - sum(errores^2) / ss_total else NA_real_,
    RMSE_2020_mas = if (any(nuevos)) sqrt(mean(errores[nuevos]^2)) else NA_real_,
    stringsAsFactors = FALSE
  )
}
resultado_test_final <- rbind(
  metricas_test("Baseline: promedio por marca", pred_baseline_test),
  metricas_test(paste0("Modelo seleccionado: ", modelo_elegido), pred_final)
)
resultado_test_final$Mejora_vs_baseline_pct <- 100 *
  (resultado_test_final$RMSE[1] - resultado_test_final$RMSE) /
  resultado_test_final$RMSE[1]

# Resultados reproducibles para el informe y la presentacion.
write.csv(resultado_test_final, "resultado_test_final.csv", row.names = FALSE)
cat("\nMODELO ELEGIDO POR VALIDACION CRUZADA:", modelo_elegido, "\n")
cat("\nRESULTADOS FINALES EN TEST (una sola evaluacion):\n")
print(resultado_test_final)
cat("\nARCHIVOS CREADOS:\n")
cat("  tabla_cv_modelos_individuales.csv\n")
cat("  tabla_comparativa_validacion.csv\n")
cat("  resultado_test_final.csv\n")
cat("  diagnostico_OLS.pdf, qq_residuos_OLS.pdf, diagnostico_GAM.pdf\n")
cat("\nTermino la ejecucion. No elegir modelos mirando el test.\n")
